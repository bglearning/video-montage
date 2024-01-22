from models.co_transformer import vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from models.co_transformer import heads, objectives#, vilt_utils

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import random

# from clip import clip

from clip_utils import text_embedder

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

class CoTransformerSM(nn.Module):
    def __init__(self, *, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.topk = args.topk
        self.only_last = False
        if hasattr(args, 'only_last'):
            self.only_last = args.only_last

        fc_input_size = (2 * args.seq_len) if self.only_last else (args.seq_len * 2 * args.seq_len)
        self.fc1 = nn.Linear(fc_input_size, args.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1.apply(objectives.init_weights)
        self.fc2 = nn.Linear(args.hidden_size, 2)
        self.fc2.apply(objectives.init_weights)
        
    
    def image2video(self, frame_embeds, text_embeds, use_key_frame = False):
        video_embeds = frame_embeds.mean(dim = 2)
        return video_embeds

    def forward(self, shots, shuffle_shots, texts, itm_labels, s_labels, text_masks, itm_masks, s_mask, image_token_type_idx = 1, return_logist = False):
        
        # texts: (n_batch, seq_len, 77)
        # shots: (n_batch, seq_len, K, 512)
        # shuffle_shots: (n_batch, seq_len, K, 512)
        # itm_labels: 64
        # s_labels: (64, 10)
        # text_masks: (64, 11)
        # itm_masks: (64, 10)
        # s_mask: (64, 10)

        batch: int = texts.shape[0]
        
        # (n_batch x seq_len, 77)
        texts = texts.reshape(-1, texts.shape[-1])

        with torch.no_grad():
            # (n_batch x seq_len, 512)
            _, text_embeds = text_embedder(texts)

        # (n_batch, seq_len, 512)
        text_embeds = text_embeds.reshape(batch, -1, text_embeds.shape[-1]).float()
        
        # itm task
        frame_embeds = shots

        # use key frame
        # (n_batch, seq_len, 512)
        video_embeds = self.image2video(frame_embeds, text_embeds)

        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
        video_embeds = torch.nn.functional.normalize(video_embeds, dim=-1)

        all_embeds = torch.cat([text_embeds, video_embeds], dim=1)
        # (n_batch, 2 x seq_len)
        co_masks = torch.cat([text_masks, itm_masks], dim=1)

        # (n_batch, 2 x seq_len, embed_dim) x (n_batch, embed_dim, 2 x seq_len)
        # -> (n_batch, 2 x seq_len, 2 x seq_len)
        sim_matrix = torch.bmm(all_embeds, all_embeds.transpose(1, 2))

        # col_i <- mask_i; so features are row-wise
        sim_matrix = torch.einsum('bij,bj->bij', sim_matrix, co_masks)

        if self.only_last:
            num_vids = itm_masks.sum(axis=1)
            vid_indices = (
                (self.seq_len + num_vids - 1)
                .reshape(-1, 1)
                .repeat(1, 2 * self.seq_len)
                .unsqueeze(1)
                .long()
            )
            # n_batch, 1, 2 * seq_len
            vid_sim_feats = sim_matrix.gather(dim=1, index=vid_indices)
        else:
            # Get {all batches}, {only vid rows}, {all columns}
            vid_sim_feats = sim_matrix[:, self.seq_len:, :]

        vid_feats_flattened = torch.flatten(vid_sim_feats, start_dim=1)
        x = self.fc1(vid_feats_flattened)
        x = self.activation(x)
        itm_logits = self.fc2(x)

        if return_logist : 
            return itm_logits, torch.tensor(0.)
        
        # itm_labels.sum() == 32
        itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
        return itm_loss, torch.tensor(0.) 

    def embed_texts(self, texts, text_masks):
        batch = texts.shape[0]
        
        texts = texts.reshape(-1, texts.shape[-1])

        with torch.no_grad():
            _, text_embeds = text_embedder(texts)

        text_embeds = text_embeds.reshape(batch, -1, text_embeds.shape[-1]).float()

        return text_embeds, text_masks


    def inference(self, shots, text_embeds, text_masks, itm_masks, image_token_type_idx = 1):
        
        # itm task
        frame_embeds = shots

        # use key frame
        # (n_batch, seq_len, 512)
        video_embeds = self.image2video(frame_embeds, text_embeds)

        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
        video_embeds = torch.nn.functional.normalize(video_embeds, dim=-1)

        all_embeds = torch.cat([text_embeds, video_embeds], dim=1)
        # (n_batch, 2 x seq_len)
        co_masks = torch.cat([text_masks, itm_masks], dim=1)

        # (n_batch, 2 x seq_len, embed_dim) x (n_batch, embed_dim, 2 x seq_len)
        # -> (n_batch, 2 x seq_len, 2 x seq_len)
        sim_matrix = torch.bmm(all_embeds, all_embeds.transpose(1, 2))

        # col_i <- mask_i; so features are row-wise
        sim_matrix = torch.einsum('bij,bj->bij', sim_matrix, co_masks)
        if self.only_last:
            num_vids = itm_masks.sum(axis=1)
            vid_indices = (
                (self.seq_len + num_vids - 1)
                .reshape(-1, 1)
                .repeat(1, 2 * self.seq_len)
                .unsqueeze(1)
                .long()
            )
            # n_batch, 1, 2 * seq_len
            vid_sim_feats = sim_matrix.gather(dim=1, index=vid_indices)
        else:
            # Get {all batches}, {only vid rows}, {all columns}
            vid_sim_feats = sim_matrix[:, self.seq_len:, :]

        vid_feats_flattened = torch.flatten(vid_sim_feats, start_dim=1)

        x = self.fc1(vid_feats_flattened)
        x = self.activation(x)
        x = self.dropout(x)
        itm_logits = self.fc2(x)
        
        return itm_logits 
        