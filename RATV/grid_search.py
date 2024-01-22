"""Searches for good threshold and weight value.

From the output file of generate.py, we can try out different
weight and threshold values.
"""
import sys
import json

from typing import Tuple

import pandas as pd


from eval_utils import intersection_over_union


def score_preds(base_preds: list, targets: list, threshold: float, weight: float) -> Tuple[float, int]:
    iou = 0.
    lens = 0
    new_preds = []
    for pred, target in zip(base_preds, targets):
        i = 1
        for pred_score in pred['scores'][1:]:
            score = pred_score['sc_m'] + pred_score['sc_s'] * weight
            # score = pred_score['sc_s']
            if score <= threshold:
                break
            i += 1

        lens += i
        new_preds.append({'caption': pred['caption'], 'shots': pred['shots'][:i]})
        
        iou += intersection_over_union(set(pred['shots'][:i]), set(target['shots']))
    
    return iou / len(base_preds), lens / len(base_preds), new_preds


def main():
    if len(sys.argv) < 3:
        raise ValueError("Invocation should be: grid_search.py <target-file> <generate-py-output> [<output-template> <threshold> <weight>]")

    target_jsonl_file = str(sys.argv[1])
    targets = []
    with open(target_jsonl_file) as f:
        for l in f.readlines():
            targets.append(json.loads(l))

    score_jsonl_file = str(sys.argv[2])

    preds = []
    with open(score_jsonl_file) as f:
        for l in f.readlines():
            preds.append(json.loads(l))

    candidates = (
        [(i / 100, 0.5) for i in range(5, 120, 5)]
        + [(i / 100, 0.75) for i in range(5, 120, 5)]
        + [(i / 100, 1.) for i in range(5, 120, 5)]
    )

    results = []

    for t, w in candidates:
        iou, avg_len, _ = score_preds(preds, targets, threshold=t, weight=w)
        results.append((t, w, iou, avg_len))

    result_df = pd.DataFrame(results, columns=['Threshold', 'weight', 'iou', 'avg_len'])

    print(result_df.sort_values(by='iou', ascending=False).head(30))

    if len(sys.argv) > 3:
        out_template = sys.argv[3]
        if len(sys.argv) > 4:
            threshold = float(sys.argv[4])
            weight = float(sys.argv[5])
        else:
            best_combination = result_df.sort_values(by='iou', ascending=False).head(1).iloc[0]
            threshold = best_combination['Threshold']
            weight = best_combination['weight']

        iou, avg_len, new_preds = score_preds(
            preds,
            targets,
            threshold=threshold,
            weight=weight,
        )

        output_file = (
            f"{out_template}"
            f"_{int(threshold * 100)}-{int(weight * 100)}"
            "-post.jsonl"
        )

        with open(output_file, 'w') as f:
            for pred in new_preds:
                f.write(f'{json.dumps(pred)}\n')
        print(f'Output with setting {threshold=}, {weight=} saved to: {output_file}')


if __name__ == '__main__':
    main()
