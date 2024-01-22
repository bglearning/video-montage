import argparse
from pathlib import Path
import heapq
from torch.backends import cudnn
import random
import numpy as np
import json
import jsonlines

import torch
import torch.nn.functional as F


from datasets.tiktok import TiktokDataset, TiktokDatasetCaps
from datasets.vspd import VSPDDataset

from models import distributed_utils

# transformer model
from models import CoTransformer

from tqdm import tqdm

from clip_utils import text_processor, text_embedder
from frames import FrameReader

from similarities import SIMILARITIES

from datasets.tiktok import load_captions


# argument parsing

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument('--model_load_path', default='', type=str,
                   help='path to your trained Transformer')

group.add_argument('--transformer_path', type=str,
                   help='path to your partially trained Transformer')

parser.add_argument('--data_dir', type=str, required=True,
                    help='path to your folder of frames')

parser.add_argument('--json_file', type=str, required=True,
                    help='path to your json file of captions and shots')
parser.add_argument('--caption_file', type=str, required=False,
                    help='Path to the corresponding captions file')
parser.add_argument('--dataset_type', type=str, default='vspd', required=False, choices=('vspd', 'tiktok', 'tiktok_caps'),
                    help='Type of dataset')

parser.add_argument('--output_file', type=str, required=True,
                    help='path to save results')

parser.add_argument('--seed', type=int, default=42, help='Seed for random number')

parser.add_argument('--target_file', type=str, default = 'custom')
parser.add_argument('--frame_suffix', type=str, default = '')

parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--batch_size', default = 16, type = int, help = 'Batch size')

train_group.add_argument("--seq_len", type=int, default=10, help="Max length of sequence")

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--clip_model', default = "ViT-B/32", type = str, help = 'Name of CLIP')

model_group.add_argument('--hidden_size', default = 512, type = int, help = 'Model dimension')

model_group.add_argument('--image_size', default = 256, type = int, help = 'Size of image')

model_group.add_argument('--num_heads', default = 8, type = int, help = 'Model number of heads')

model_group.add_argument('--num_layers', default = 2, type = int, help = 'Model depth')

model_group.add_argument('--topk', default = 4, type = int)

model_group.add_argument('--threshold', default = 0.9, type = float)

model_group.add_argument('--weight', default = 1.0, type = float)
model_group.add_argument('--model_score_weight', default = 1.0, type = float)
model_group.add_argument('--allow_multi_embed', action='store_true')
model_group.add_argument('--sim_func', default = "default", type = str, help = 'Name of similarity function')

args = parser.parse_args()

# random seed
cudnn.benchmark = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)

seq_len = args.seq_len

# helper fns

def exists(val):
    return val is not None

def check_length(sequence, mask) : 
    assert isinstance(sequence,list)
    
    if len(sequence) >= seq_len : 
        sequence = sequence[:seq_len]
    while len(sequence) < seq_len : 
        empty = torch.zeros_like(sequence[0])
        sequence.append(empty)
        mask[len(sequence) - 1] = 0
    
    sequence = torch.stack(sequence, dim = 0)
    
    return sequence, mask
    

# Only load the model if it needs to be used
if args.model_score_weight > 0:
    TRANSFORMER_PATH = args.transformer_path

    assert Path(TRANSFORMER_PATH).exists(), 'trained model must exist'

    loaded_obj = torch.load(str(TRANSFORMER_PATH), map_location='cpu')

    transformer_params, weights = loaded_obj['hparams'], loaded_obj['weights']

    transformer_params = dict(
        **transformer_params
    )

    transformer = CoTransformer(**transformer_params)

    transformer = transformer.cuda()

    transformer.load_state_dict(weights)
    transformer.eval()

frame_reader = FrameReader(data_dir=args.data_dir,
                           suffix=args.frame_suffix,
                           k=8,
                           embed_single=(not args.allow_multi_embed))

instances = []
with open(args.json_file, "r", encoding="utf8") as f:
    for line in f.readlines():
        instances.append(json.loads(line))

captions = (
    load_captions(args.caption_file) if args.caption_file is not None else None
)
dataset_type = args.dataset_type if hasattr(args, 'dataset_type') else 'vspd'

if dataset_type == 'tiktok':
    dataset = TiktokDataset(instances, captions)
elif dataset_type == 'tiktok_caps':
    dataset = TiktokDatasetCaps(instances, captions)
elif dataset_type == 'vspd':
    dataset = VSPDDataset(instances, captions)

# get dataset
texts = []
shots = []
shots_proj = []
shot_names = []
gt_shots = []

for ind in range(len(dataset)):
    item = dataset[ind]
    texts.append(dataset.get_text(ind))
    shot_names.extend(item['shots'])
    embed, embed_proj = frame_reader.read(item['shots'])
    shots.extend(embed)
    shots_proj.extend(embed_proj)
        
    gt_shots.append(item['shots'])


print(f'Loaded {args.json_file}: {len(texts)}')

print(f'Sim func: {args.sim_func}')
sim_func = SIMILARITIES.get(args.sim_func)

# generate videos
scores_output_file = str(args.output_file).replace(".jsonl", "-scores.jsonl")

for j, sentences in tqdm(enumerate(texts)) : 
    text_infos = []
    
    for sentence in sentences : 
        text_infos.append(text_processor(sentence).squeeze(0))
    
    text_clip = torch.stack(text_infos, dim = 0).cuda()
    with torch.no_grad():
        text_embeds_proj, text_embeds = text_embedder(text_clip)
    text_embeds = torch.mean(text_embeds, dim = 0).unsqueeze(0).float()
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
    text_embeds_proj = torch.mean(text_embeds_proj, dim = 0).unsqueeze(0).float()
    text_embeds_proj = torch.nn.functional.normalize(text_embeds_proj, dim=-1)
    
    text_mask = torch.ones([seq_len], dtype = torch.float)
    
    captions, text_mask = check_length(text_infos, text_mask)
    captions = captions.unsqueeze(dim = 0).cuda()
    text_mask = text_mask.unsqueeze(dim = 0).cuda()

    text_embeds_, text_masks = None, None
    
    if args.model_score_weight > 0:
        text_embeds_, text_masks = transformer.embed_texts(captions, text_mask)
    
    threshold = args.threshold
    chosen_shots = []
    chosen_shot_scores = []
    shot_list = []

    all_shot_scores = []
    
    while True:
        if len(chosen_shots) >= args.seq_len:
            break

        top_k_candidates = []
        TOP_K = 50

        # We first retrieve the top-k best candidates only based on sim-score

        for i in range(len(shots_proj)):
            if shot_names[i] in chosen_shots:
                continue

            sim = sim_func(shots_proj[i], text_embeds_proj.cpu())
            entry_tuple = (sim, i)

            if len(top_k_candidates) < TOP_K:
                heapq.heappush(top_k_candidates, entry_tuple)
            else:
                heapq.heappushpop(top_k_candidates, entry_tuple)

        max_l = {'sc_m': 0., 'sc_s': 0., 'tot': 0.}
        max_index = 0

        # Now rank from among the top-k candidates
        for sim, i in top_k_candidates: 
            if shot_names[i] in chosen_shots: 
                continue

            model_score = 0.

            if args.model_score_weight > 0:
                shot = shots[i]
                shot_mask = torch.ones([args.seq_len], dtype = torch.float)
                shot, shot_mask = check_length(shot_list + [shot], shot_mask)
                shot = shot.unsqueeze(dim = 0).cuda()
                shot_mask = shot_mask.unsqueeze(dim = 0).cuda()
                
                logits = transformer.inference(shot, text_embeds_, text_masks, shot_mask)
                logits = F.softmax(logits, dim = -1)
                
                model_score = logits[0, 1].item()

            score = model_score * args.model_score_weight + args.weight * sim
            all_shot_scores.append((j, i, model_score, sim))
            
            if score > max_l['tot']:
                max_l = {'sc_m': model_score, 'sc_s': sim, 'tot': score}
                max_index = i
        
        # Stop if max score doesn't the threshold unless we haven't even selected one shot
        if (len(chosen_shots) != 0) and (max_l['tot'] <= threshold):
            break
        # Add chosen shot to the list to be prepended to the candidate in the next pass
        shot_list.append(shots[max_index])
        chosen_shots.append(shot_names[max_index])
        chosen_shot_scores.append(max_l)
    
    item = {}
    item['caption'] = '. '.join(sentences)
    item['shots'] = chosen_shots
    item['scores'] = chosen_shot_scores
    
    with jsonlines.open(args.output_file, mode='a') as f :
        f.write(item)

    # For statistics
    with jsonlines.open(scores_output_file, mode='a') as f :
        f.write(all_shot_scores)
    