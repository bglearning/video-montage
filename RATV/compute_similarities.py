import json
import sys

import numpy as np
import torch

from sklearn.metrics import pairwise_distances

from frames import FrameReader


def cosine_similarity_matrix(embeddings):
    similarity_matrix = 1 - pairwise_distances(embeddings, metric='cosine')
    return similarity_matrix


def main():

    json_file = sys.argv[1]
    data_dir = sys.argv[2]
    out_file = sys.argv[3]

    frame_reader = FrameReader(data_dir=data_dir, suffix='', k=8, embed_single=True)

    embeddings = []

    with open(json_file, "r", encoding="utf8") as f:
        i = 0
        for l in f.readlines():
            _, shot_projs = frame_reader.read(json.loads(l)['shots'])
            # Mean across frames of the same shot, then mean across all the shots
            embeddings.append(torch.stack(shot_projs).mean(axis=1).mean(axis=0))
            i += 1
            if i % 500 == 0:
                print(f'{i} instances processed')

    embeddings = torch.stack(embeddings)
    print(embeddings.shape)
    sims = cosine_similarity_matrix(embeddings.detach().numpy())
    print(sims.shape)
    np.save(out_file, sims)


if __name__ == '__main__':
    main()