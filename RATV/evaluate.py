import json
import os
import jsonlines
import argparse
import numpy as np
# from clip import clip
import torch

from clip_utils import text_embedder, text_processor
from frames import FrameReader

parser = argparse.ArgumentParser()

parser.add_argument('--target_file', type=str, default = 'dataset/test_captions.jsonl')

parser.add_argument('--generated_file', type=str, required=True)

parser.add_argument('--clip_model', default = "ViT-B/32", type = str, help = 'Name of CLIP')

parser.add_argument('--fea_root', default = "dataset/frame_fea", type = str, help = 'path of features')

parser.add_argument('--topk', type=int, default=1)

parser.add_argument('--frame_suffix', type=str, default = '')

args = parser.parse_args()


frame_fea_root = args.fea_root

gt = {}

gt_items = []

with open(args.target_file, "r", encoding="utf8") as f:
    for l in f.readlines():
        gt_items.append(json.loads(l))

results = []

with open(args.generated_file, "r", encoding="utf8") as f:
    for item, gt_item in zip(jsonlines.Reader(f), gt_items): 
        gt[item['caption']] = gt_item['shots']
        results.append(item)

# clip_model, _ = clip.load(args.clip_model, jit=False)
# clip_model.eval()
# clip_model = clip_model.cuda()


final_score = 0
temporal_score = 0
residual_sim = 0
avg_unmatched_dsim = 0
false_num = 0
R_recall = 0
R_recall_gt = 0
avg_len = 0

print("len(results) : ", len(results))

shot_feas = []

K = 8

frame_reader = FrameReader(data_dir=frame_fea_root,
                           suffix=args.frame_suffix,
                           k=K)

for result in results : 
    _, embeds_proj = frame_reader.read(result['shots'])
    shot_feas.append(embeds_proj) 


max_score = max_index = 0


counter = 0

for idx, (result) in enumerate(results) : 
    text = result['caption']
    chosen_shots = result['shots']
    avg_len += len(chosen_shots)
    gt_shots = gt[text]
    
    # encode text 
    sentences = text.strip().split('.')

    sentences = [s for s in sentences if s != '']

    for k in range(len(sentences)) : 
        sentences[k] = sentences[k] + '.'
    
    text_infos = []
    
    for sentence in sentences : 
        text_infos.append(text_processor(sentence).squeeze(0))
    
    text_infos = torch.stack(text_infos, dim = 0).cuda()

    with torch.no_grad():
        text_embeds, _ = text_embedder(text_infos)
        text_embeds = text_embeds.float().cpu()
    
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
    
    # IoU
    union_set = list(set(chosen_shots + gt_shots))
    union_length = len(union_set)
    
    num = 0
    orders = []
    unmatched_dsim = 0
    for n, (shot) in enumerate(chosen_shots) : 
        if shot in gt_shots : 
            num += 1
            orders.append(gt_shots.index(shot))
        else : 
            shot_embeds = shot_feas[idx][n].float().cpu()
            shot_embeds = torch.mean(shot_embeds, dim = 0).unsqueeze(0)
            shot_embeds = torch.nn.functional.normalize(shot_embeds, dim=-1)
            sim = text_embeds @ shot_embeds.T
            sim = sim[sim.argmax(dim = 0)]

            residual_sim += (1 - sim)
            unmatched_dsim += (1 - sim)
            false_num += 1

    final_score += num / union_length
    avg_unmatched_dsim += unmatched_dsim / len(chosen_shots)
    
    # temporal
    num = 0
    for i in range(len(gt_shots)) :
        if i >= len(chosen_shots) : 
            break
        elif chosen_shots[i] == gt_shots[i] : 
            num += 1
    temporal_score += num / len(gt_shots)
    
    num = max_num = 0
    for o, (order) in enumerate(orders) : 
        if o == len(orders) - 1 : 
            if num > max_num : 
                max_num = num
            break
        
        if order + 1 == orders[o+1] :
            num += 1
        else : 
            if num > max_num : 
                max_num = num
            num = 0
    
    counter += 1

final_score = final_score / counter
temporal_score = temporal_score / counter
residual_sim = residual_sim / counter
avg_unmatched_dsim = avg_unmatched_dsim / counter

print("counter : ", counter)
print("IoU : ", final_score)
print("temporal_score : ", temporal_score)
print("residual_sim : ", residual_sim)
print("avg_unmatched_dsim : ", avg_unmatched_dsim)
print("avg_len : ", avg_len / len(results))