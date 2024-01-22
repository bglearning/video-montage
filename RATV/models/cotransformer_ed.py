from models.co_transformer import vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from models.co_transformer import heads, objectives#, vilt_utils

import torch.nn as nn
import torch
import torch.nn.functional as F

from clip_utils import text_embedder


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class CoTransformerED(nn.Module):
    def __init__(self, *, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.topk = args.topk

        self.transformer = nn.Transformer(
            d_model=args.hidden_size,
            nhead=args.num_heads,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            dim_feedforward=args.hidden_size,
            batch_first=True,
        )

        self.enc_pos_emb = nn.Embedding(args.seq_len, args.hidden_size)
        self.enc_pos_emb.apply(objectives.init_weights)

        self.dec_pos_emb = nn.Embedding(args.seq_len + 1, args.hidden_size)
        self.dec_pos_emb.apply(objectives.init_weights)
        
        self.cls_token = nn.Parameter(torch.zeros(args.hidden_size))
        
        self.pooler = heads.Pooler(args.hidden_size)
        self.pooler.apply(objectives.init_weights)
        
        self.itm_score = heads.ITMHead(args.hidden_size)
        self.itm_score.apply(objectives.init_weights)
        
    
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

        # (n_batch, 11)
        # text_masks = torch.cat([torch.ones(batch, 1).cuda(), text_masks], dim = 1)

        with torch.no_grad():
            # (n_batch x seq_len, 512)
            _, text_embeds = text_embedder(texts)

        # (n_batch, seq_len, 512)
        text_embeds = text_embeds.reshape(batch, -1, text_embeds.shape[-1]).float()
        
        # (n_batch, 1, 512)
        # cls_embeds = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(text_embeds.shape[0], 1, 1)
        
        # (n_batch, seq_len + 1, 512)
        # text_embeds = torch.cat([cls_embeds, text_embeds], dim = 1)
        
        text_embeds = text_embeds + self.enc_pos_emb(torch.arange(text_embeds.shape[1]).cuda())
        
        # itm task
        frame_embeds = shots

        # use key frame
        # (n_batch, seq_len, 512)
        video_embeds = self.image2video(frame_embeds, text_embeds)
        
        # (n_batch, 1, 512)
        cls_embeds = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(video_embeds.shape[0], 1, 1)
        
        # (n_batch, seq_len + 1, 512)
        video_embeds = torch.cat([cls_embeds, video_embeds], dim = 1)

        video_embeds = video_embeds + self.dec_pos_emb(torch.arange(video_embeds.shape[1]).cuda())

        out = self.transformer(
            src=text_embeds,
            tgt=video_embeds,
            src_key_padding_mask=text_masks,
            # tgt_key_padding_mask=s_mask,
        )
        
        # (n_batch, 512)
        cls_feats = self.pooler(out)
        
        # (n_batch, 2)
        itm_logits = self.itm_score(cls_feats)
        
        if return_logist : 
            return itm_logits, 0.
        
        # itm_labels.sum() == 32
        itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

        return itm_loss, 0. 

    def embed_texts(self, texts, text_masks):
        batch: int = texts.shape[0]
        
        # (n_batch x seq_len, 77)
        texts = texts.reshape(-1, texts.shape[-1])

        # (n_batch, 11)
        # text_masks = torch.cat([torch.ones(batch, 1).cuda(), text_masks], dim = 1)

        with torch.no_grad():
            # (n_batch x seq_len, 512)
            _, text_embeds = text_embedder(texts)

        # (n_batch, seq_len, 512)
        text_embeds = text_embeds.reshape(batch, -1, text_embeds.shape[-1]).float()
        
        # (n_batch, 1, 512)
        # cls_embeds = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(text_embeds.shape[0], 1, 1)
        
        # (n_batch, seq_len + 1, 512)
        # text_embeds = torch.cat([cls_embeds, text_embeds], dim = 1)
        
        text_embeds = text_embeds + self.enc_pos_emb(torch.arange(text_embeds.shape[1]).cuda())

        return text_embeds, text_masks


    def inference(self, shots, text_embeds, text_masks, itm_masks, image_token_type_idx = 1):
        # itm task
        frame_embeds = shots

        # use key frame
        # (n_batch, seq_len, 512)
        video_embeds = self.image2video(frame_embeds, text_embeds)
        
        # (n_batch, 1, 512)
        cls_embeds = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(video_embeds.shape[0], 1, 1)
        
        # (n_batch, seq_len + 1, 512)
        video_embeds = torch.cat([cls_embeds, video_embeds], dim = 1)

        video_embeds = video_embeds + self.dec_pos_emb(torch.arange(video_embeds.shape[1]).cuda())

        out = self.transformer(
            src=text_embeds,
            tgt=video_embeds,
            src_key_padding_mask=text_masks,
            # tgt_key_padding_mask=itm_masks,
        )
        
        # (n_batch, 512)
        cls_feats = self.pooler(out)
        
        # (n_batch, 2)
        itm_logits = self.itm_score(cls_feats)
        
        return itm_logits 
        