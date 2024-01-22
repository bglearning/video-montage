import json
import random
import sys

from pathlib import Path


def main():
    random.seed(0)

    output_dir = Path.cwd()

    if len(sys.argv) == 2:
        output_dir = Path(sys.argv[1])
        output_dir.mkdir(exist_ok=True)

    with open("tiktok_montage_gt_4w_9k_emb.json") as f:
        data = json.load(f)

    print("Total data: ", len(data))
    random.shuffle(data)

    TRAIN, VAL, TEST = 7896, 500, 500

    with open(output_dir / "tiktok_montage_v1_train.jsonl", "w") as f:
        for d in data[:TRAIN]:
            f.write(json.dumps(d) + '\n')

    with open(output_dir / "tiktok_montage_v1_val.jsonl", "w") as f:
        for d in data[TRAIN:TRAIN + VAL]:
            f.write(json.dumps(d) + '\n')

    with open(output_dir / "tiktok_montage_v1_test.jsonl", "w") as f:
        for d in data[TRAIN + VAL:TRAIN + VAL + TEST]:
            f.write(json.dumps(d) + '\n')

if __name__ == '__main__':
    main()
