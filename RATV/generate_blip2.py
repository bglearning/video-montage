import sys

from pathlib import Path
from typing import List

import json

import jsonlines
import numpy as np
import torch

from tqdm import tqdm

from clip_video_encode.encoders import create_encoder


def main():
    test_caption_file = Path(sys.argv[1])
    vid_embeddings_dir = Path(sys.argv[2])
    text_embeddings_dir = Path(sys.argv[3])
    threshold = float(sys.argv[4])
    output_file = Path(sys.argv[5])

    MAX_SEQ_LEN = 10

    test_cases = []
    texts = []
    shot_names = []
    shots = []
    gt_shots = []

    with open(test_caption_file) as f:
        for l in f.readlines():
            test_case = json.loads(l)
            texts.append(test_case['caption'])
            test_cases.append(test_case)

            for shot in test_case['shots'] : 
                shot_names.append(shot)
                # frame_path = os.path.join(args.data_dir, shot, 'fea.npy')
                shot_num = shot.split(".")[0]
                frame_path = vid_embeddings_dir / f'{shot_num}.npy'
                
                # (n_frames, 32, 768)
                frames_fea = np.load(frame_path)

                # (n_frames, 768)
                frames_fea = frames_fea.mean(axis=1)
                # (768,)
                frames_fea = frames_fea.mean(axis=0)

                frames_fea = frames_fea.astype(np.float32)
                frames_fea = torch.from_numpy(frames_fea)
                
                shots.append(frames_fea)
                
            gt_shots.append(test_case['shots'])

    text_embeddings = []

    if text_embeddings_dir.exists():
        print("Loading text embeddings...")
        for i in range(len(texts)):
            text_embeddings.append(np.load(text_embeddings_dir / f'{i}.npy'))
    else:
        print("Creating text embeddings...")
        blip2_encoder = create_encoder("blip2")

        for i, text in enumerate(texts):
            sentences = text.strip().split('.')

            sentences = [s for s in sentences if s != '']
            
            for k in range(len(sentences)) : 
                sentences[k] = sentences[k] + '.'
            
            # (n_sentences, 7, 768)
            s_encodings = blip2_encoder.encode_captions(sentences)
            # (n_sentences, 768)
            s_encodings = s_encodings.mean(axis=1)
            # (768,)
            text_encoding = s_encodings.mean(axis=0)
            text_encoding = text_encoding.astype(np.float32)
            np.save(text_embeddings_dir / f'{i}.npy', text_encoding)
            text_embeddings.append(text_encoding)

    for j, (text) in tqdm(enumerate(texts)) : 
        text_embed = torch.from_numpy(text_embeddings[j])
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
        
        chosen_shots = []
        chosen_shot_scores = []
        
        while True:
            if len(chosen_shots) >= MAX_SEQ_LEN:
                break

            max_l = {'sc_m': 0., 'sc_s': 0., 'tot': 0.}
            max_index = 0
            for i, (shot) in enumerate(shots): 
                if shot_names[i] in chosen_shots: 
                    continue
                
                shot_embed = torch.nn.functional.normalize(shot, dim=-1)
                
                # score = torch.mm(text_embeds.cpu(), shot_embeds.T).float()[0][0].item()
                score = torch.dot(text_embed, shot_embed).item()
                
                if score > max_l['tot']:
                    max_l = {'tot': score}
                    max_index = i
            
            # Stop if max score doesn't the threshold unless we haven't even selected one shot
            if (len(chosen_shots) != 0) and (max_l['tot'] <= threshold):
                break
            chosen_shots.append(shot_names[max_index])
            chosen_shot_scores.append(max_l)
        
        item = {}
        item['caption'] = text
        item['shots'] = chosen_shots
        item['scores'] = chosen_shot_scores
        
        # for statistic
        with jsonlines.open(output_file, mode='a') as f :
            f.write(item)


if __name__ == '__main__':
    main()