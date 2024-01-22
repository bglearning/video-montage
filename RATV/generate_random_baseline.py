import json
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        raise ValueError("Usage: <script> <test-file.jsonl> <output-file.jsonl>")

    test_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    texts = []
    shot_names = []

    with open(test_file_path, "r", encoding="utf8") as f:
        for l in f.readlines(): 
            item = json.loads(l)
            texts.append(item['caption'])
            for shot in item['shots'] : 
                shot_names.append(shot)

    print(f"Loaded file: {test_file_path}, n: {len(texts)}, num_candidate_shots: {len(shot_names)}")

    np.random.seed(1)

    for text in texts:
        item = {}
        item['caption'] = text

        num_shots: int = np.random.choice(np.arange(1, 11), 1)[0]
        chosen_shot_indices = np.random.choice(len(shot_names), num_shots, replace=False)
        item['shots'] = [shot_names[i] for i in chosen_shot_indices]
        
        with open(output_file_path, mode='a') as f :
            f.write(f'{json.dumps(item)}\n')
        
        
if __name__ == '__main__':
    main()