import random

import numpy as np
import torch


def check_length(sequence, mask, seq_len) : 
    assert isinstance(sequence,torch.Tensor)
    
    if sequence.shape[0] >= seq_len : 
        sequence = sequence[:seq_len]
        mask = mask[:seq_len]
    while sequence.shape[0] < seq_len : 
        empty = torch.zeros_like(sequence[0]).unsqueeze(dim = 0)
        empty_mask = torch.zeros_like(mask[0]).unsqueeze(dim = 0)
        sequence = torch.cat([sequence, empty], dim = 0)
        mask = torch.cat([mask, empty_mask], dim = 0)
    
    return sequence, mask


def neg_slice_insert(shot, s_mask, false_shot, f_mask, seq_len):
    # random select position and length to replace
    l = s_mask.sum(dim = -1).int().item()
    begin_pos = random.randint(0, l)
    
    if l == seq_len and l == begin_pos: 
        begin_pos = begin_pos - random.randint(1, l)
    
    l_f = f_mask.sum(dim = -1).int().item()
    if l_f == 1 : 
        f_pos = 0
    else :
        f_pos = random.randint(0, l_f - 1)
    if l_f - f_pos <= 1: 
        f_length = 1
    else : 
        f_l = min(l_f - f_pos, int(l_f / 2))
        f_length = random.randint(1, f_l)
    
    input_shots = torch.cat([shot[:begin_pos], false_shot[f_pos: f_pos + f_length], shot[begin_pos:]], dim = 0)
    input_mask = torch.cat([s_mask[:begin_pos], f_mask[f_pos: f_pos + f_length], s_mask[begin_pos:]], dim = 0)
    return input_shots, input_mask
                

def neg_next_shot(shot, s_mask, false_shot, f_mask, seq_len):
    # Next single shot negatives
    # random select position and length to replace
    l = s_mask.sum(dim = -1).int().item()
    probs = np.array([1./l] * l)
    probs[0] = probs[0] + 0.2
    probs = probs / probs.sum()
    
    begin_pos = np.random.choice(l, p=probs)
    
    if l == seq_len and l == begin_pos: 
        begin_pos = begin_pos - random.randint(1, l)
    
    l_f = f_mask.sum(dim = -1).int().item()
    if l_f == 1 : 
        f_pos = 0
    else :
        f_pos = random.randint(0, l_f - 1)
    f_length = 1
    
    input_shots = torch.cat([shot[:begin_pos], false_shot[f_pos: f_pos + f_length]], dim = 0)
    input_mask = torch.cat([s_mask[:begin_pos], f_mask[f_pos: f_pos + f_length]], dim = 0)
    return input_shots, input_mask


class SequenceMatching:
    def __init__(self, seq_len: int, negatives: str = 'slice_insert', pos_ratio: float = 0.5) -> None:
        self.seq_len = seq_len
        if negatives == 'slice_insert':
            self.neg_func = neg_slice_insert
        elif negatives == 'next_shot':
            self.neg_func = neg_next_shot
        self.pos_ratio = pos_ratio

    def create(self, shots, shot_mask, false_shots, false_mask):
        # create text-image pairs for itm task
        assert shots.shape[0] == false_shots.shape[0] and shots.shape[0] > 2
        
        length = shots.shape[0]
        
        pos_len = int(length * self.pos_ratio)
        neg_len = length - pos_len
        
        itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).cuda()
        
        itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
        
        itm_shots = []
        itm_mask = []
        for idx in range(length): 
            input_shots = shots[idx]
            input_mask = shot_mask[idx]
            if itm_labels[idx] == 1 : 
                l = input_mask.sum(dim = -1).int().item()
                
                m_length = np.random.choice(l)
                m_range = torch.arange(input_mask.shape[-1] - m_length) + m_length

                input_mask[m_range] = 0
                
            elif itm_labels[idx] == 0:
                input_shots, input_mask = self.neg_func(input_shots, input_mask, false_shots[idx], false_mask[idx],
                                                        seq_len=self.seq_len)
                input_shots, input_mask = check_length(input_shots, input_mask, self.seq_len)
            else : 
                print("label error in itm")
                exit()
            
            itm_shots.append(input_shots)
            itm_mask.append(input_mask)
        
        itm_shots = torch.stack(itm_shots).cuda()
        itm_mask = torch.stack(itm_mask).cuda()

        return itm_shots, itm_mask, itm_labels
        

class SequenceRecovery:
    def __init__(self) -> None:
        pass

    def create(self, shots, shot_mask):

        length = shots.shape[0]
        shuffle_shots = []
        s_mask = []
        s_labels = []
        
        for idx in range(length) : 
            input_mask = shot_mask[idx]
            l = input_mask.sum(dim = -1).int().item()

            s_range = torch.randperm(l)
            l_range = torch.arange(input_mask.shape[-1] - l, dtype=torch.long) + l#.long()
            n_range = torch.cat([s_range, l_range])

            input_shots = shots[idx][n_range]
            
            label = n_range
            
            # random mask
            label[~input_mask.bool()] = -100

            shuffle_shots.append(input_shots)
            s_mask.append(input_mask)
            s_labels.append(label)
        
        shuffle_shots = torch.stack(shuffle_shots).cuda()
        s_mask = torch.stack(s_mask).cuda()
        s_labels = torch.stack(s_labels).cuda()

        return shuffle_shots, s_mask, s_labels
        