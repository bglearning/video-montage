import json

from torchvision import transforms
from PIL import Image
from PIL import ImageFile

from collections import defaultdict
from typing import Dict, List


ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
from typing import List
import numpy as np
import torch
import torch.utils.data as data
import jsonlines
import random
import time
# from clip import clip
import numpy as np

from clip_utils import text_processor
from frames import FrameReader

from datasets.tiktok import load_captions, TiktokDataset, TiktokDatasetCaps
from datasets.vspd import VSPDDataset


def top_k_indices(similarity_matrix, k):
    # Use argpartition to get the indices of the k smallest elements for each row
    # We use "-k" since argpartition returns the indices of the k smallest elements
    top_k_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]
    return top_k_indices


def bottom_k_indices(similarity_matrix, k):
    bottom_k_ind = np.argpartition(similarity_matrix, k, axis=1)[:, :k]
    return bottom_k_ind


class NegSamplerFull:
    def sample(self, num_identifiers: int, pos_index: int) -> int:
        random_index = random.randint(0, num_identifiers - 1)
        while random_index == pos_index: 
            random_index = random.randint(0, num_identifiers - 1)
        return random_index


class NegSamplerRandomThenSim:
    def __init__(self, sim_matrix: np.array, random_sample_size: int = 20) -> None:
        self.sim_matrix = sim_matrix
        self.random_sample_size = random_sample_size

    def sample(self, num_identifiers: int, pos_index: int) -> int:
        candidate_indices = np.random.randint(0, num_identifiers - 1, size=self.random_sample_size)
        candidate_indices = candidate_indices + (candidate_indices >= pos_index).astype(int)

        candidate_sims = self.sim_matrix[pos_index][candidate_indices]
        candidate_max_i = np.argmax(candidate_sims)
        candidate_index = candidate_indices[candidate_max_i]
        return candidate_index


class NegSamplerRandomFromSim:
    def __init__(self, sim_matrix: np.array) -> None:
        self.sim_matrix = sim_matrix

    def sample(self, num_identifiers: int, pos_index: int) -> int:
        candidate_indices = self.sim_matrix[pos_index]
        random_index = random.randint(0, len(candidate_indices) - 1)
        candidate_index = candidate_indices[random_index]
        while candidate_index == pos_index : 
            random_index = random.randint(0, len(candidate_indices) - 1)
            candidate_index = candidate_indices[random_index]
        return candidate_index


class AllDataset(data.Dataset):
    def __init__(self, args):
        #assert args.phase == 'train'
        
        self.data_dir = args.data_dir
        self.seq_len = args.seq_len
        self.k = 8
        self.embed_dim = args.embed_dim
        
        self.frame_reader = FrameReader(data_dir=args.data_dir,
                                        suffix=args.frame_suffix,
                                        k=self.k)
        
        if args.neg_sampler == 'full_random':
            self.neg_sampler = NegSamplerFull()
        elif args.neg_sampler == 'random_most_sim':
            full_sim_matrix = np.load(args.similarities_file)
            self.neg_sampler = NegSamplerRandomThenSim(full_sim_matrix)
        elif args.neg_sampler == "random_from_bottom":
            full_sim_matrix = np.load(args.similarities_file)
            np.fill_diagonal(full_sim_matrix, 1.)
            bottom_sim_matrix = bottom_k_indices(full_sim_matrix, k=100)
            self.neg_sampler = NegSamplerRandomFromSim(bottom_sim_matrix)
        elif args.neg_sampler == "random_from_top":
            full_sim_matrix = np.load(args.similarities_file)
            np.fill_diagonal(full_sim_matrix, 0.)
            top_sim_matrix = top_k_indices(full_sim_matrix, k=100)
            self.neg_sampler = NegSamplerRandomFromSim(top_sim_matrix)


        json_file = args.json_file

        instances = [] 

        with open(json_file, "r", encoding="utf8") as f:
            for l in f.readlines():
                instances.append(json.loads(l))
        
        print('Load {} {}'.format(json_file, len(instances)))

        captions = (
            load_captions(args.caption_file) if args.caption_file is not None else None
        )
        dataset_type = args.dataset_type if hasattr(args, 'dataset_type') else 'vspd'

        if dataset_type == 'tiktok':
            self.dataset = TiktokDataset(instances, captions)
        elif dataset_type == 'tiktok_caps':
            self.dataset = TiktokDatasetCaps(instances, captions)
        elif dataset_type == 'vspd':
            self.dataset = VSPDDataset(instances, captions)


    def read_shots(self, shot_strs: List[str]):
        # Ignore the projections here
        shots, _ = self.frame_reader.read(shot_strs)
        
        shot_mask = torch.ones([self.seq_len], dtype = torch.float)
        
        if len(shots) >= self.seq_len : 
            shots = shots[:self.seq_len]
        if len(shots) == 0 : 
            shots.append(torch.zeros((self.k, self.embed_dim)))
        while len(shots) < self.seq_len : 
            empty = torch.zeros_like(shots[0])
            shots.append(empty)
            shot_mask[len(shots) - 1] = 0

        shots = torch.stack(shots, dim = 0)
        return shots, shot_mask

    def __getitem__(self, index):
        shots, shot_mask = self.read_shots(self.dataset[index]['shots'])
        
        # get random images for false pairs
        random_index = self.neg_sampler.sample(num_identifiers=len(self.dataset), pos_index=index)

        false_infos = self.dataset[random_index]

        false_shots, false_mask = self.read_shots(false_infos['shots'])

        # get texts 
        sentences = self.dataset.get_text(index)

        text_infos = []
        
        for sentence in sentences : 
            text_infos.append(text_processor(sentence).squeeze(0))
        
        text_mask = torch.ones([self.seq_len], dtype = torch.float)
        
        if len(text_infos) >= self.seq_len : 
            text_infos = text_infos[:self.seq_len]
        while len(text_infos) < self.seq_len : 
            empty = torch.zeros_like(text_infos[0])
            text_infos.append(empty)
            text_mask[len(text_infos) - 1] = 0
        
        text_infos = torch.stack(text_infos, dim = 0)
        
        '''
        if shots.shape[0] > 10 or text_infos.shape[0] > 10 : 
            print(shots.shape, text_infos.shape)
            #exit()
        '''
        return shots, false_shots, text_infos, shot_mask, false_mask, text_mask

    def __len__(self):
        return len(self.dataset)