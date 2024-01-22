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

class CoTransformer(nn.Module):
    def __init__(self, *, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.topk = args.topk

        # self.clip, _ = clip.load(args.clip_model, jit=False)
        # self.clip.eval()
        # set_requires_grad(self.clip, False)

        transformer_layer = nn.TransformerEncoderLayer(args.hidden_size, nhead=args.num_heads)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=args.num_layers)

        self.token_type_embeddings = nn.Embedding(2, args.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        
        self.text_pos_emb = nn.Embedding(args.seq_len + 1, args.hidden_size)
        self.shot_pos_emb = nn.Embedding(args.seq_len, args.hidden_size)
        self.text_pos_emb.apply(objectives.init_weights)
        self.shot_pos_emb.apply(objectives.init_weights)
        
        self.cls_token = nn.Parameter(torch.zeros(args.hidden_size))
        
        self.pooler = heads.Pooler(args.hidden_size)
        self.pooler.apply(objectives.init_weights)
        
        config = BertConfig(
            vocab_size=args.seq_len,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            intermediate_size=args.hidden_size * args.mlp_ratio,
            max_position_embeddings=args.seq_len,
            hidden_dropout_prob=args.drop_rate,
            attention_probs_dropout_prob=args.drop_rate,
        )
        
        self.shuffle_score = heads.MLMHead(config)
        self.shuffle_score.apply(objectives.init_weights)
        
        self.itm_score = heads.ITMHead(args.hidden_size)
        self.itm_score.apply(objectives.init_weights)
        
        # ===================== Downstream ===================== #
        if args.model_load_path != "" :
            ckpt = torch.load(args.model_load_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print("load vilt pretrained model")
    
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
        text_masks = torch.cat([torch.ones(batch, 1).cuda(), text_masks], dim = 1)

        with torch.no_grad():
            # (n_batch x seq_len, 512)
            _, text_embeds = text_embedder(texts)

        # (n_batch, seq_len, 512)
        text_embeds = text_embeds.reshape(batch, -1, text_embeds.shape[-1]).float()
        
        # (n_batch, 1, 512)
        cls_embeds = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(text_embeds.shape[0], 1, 1)
        
        # (n_batch, seq_len + 1, 512)
        text_embeds = torch.cat([cls_embeds, text_embeds], dim = 1)
        
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks, dtype = torch.long))
        
        text_embeds = text_embeds + self.text_pos_emb(torch.arange(text_embeds.shape[1]).cuda())
        
        # itm task
        frame_embeds = shots

        # use key frame
        # (n_batch, seq_len, 512)
        video_embeds = self.image2video(frame_embeds, text_embeds)
        
        video_embeds = video_embeds + self.token_type_embeddings(torch.full_like(itm_masks, image_token_type_idx, dtype = torch.long))
        
        video_embeds = video_embeds + self.shot_pos_emb(torch.arange(video_embeds.shape[1]).cuda())
        
        # (n_batch, 21, 512)
        co_embeds = torch.cat([text_embeds, video_embeds], dim=1)

        # (n_batch, 21)
        co_masks = torch.cat([text_masks, itm_masks], dim=1)
        
        co_masks = ~co_masks.bool()
    
        # (n_batch, 21, 512)
        x = self.transformer(co_embeds.transpose(0, 1), src_key_padding_mask = co_masks).transpose(0,1)

        # (n_batch, 512)
        cls_feats = self.pooler(x)
        
        # shuffle task
        frame_embeds = shuffle_shots
        
        # use key frame
        # (n_batch, seq_len, 512)
        video_embeds = self.image2video(frame_embeds, text_embeds)
        
        video_embeds = video_embeds + self.token_type_embeddings(torch.full_like(itm_masks, image_token_type_idx, dtype = torch.long))
        
        #video_embeds = video_embeds + self.shot_pos_emb(torch.arange(video_embeds.shape[1]).cuda())
        
        co_embeds = torch.cat([text_embeds, video_embeds], dim=1)
        co_masks = torch.cat([text_masks, s_mask], dim=1)
        
        co_masks = ~co_masks.bool()
    
        x = self.transformer(co_embeds.transpose(0, 1), src_key_padding_mask = co_masks).transpose(0,1)
        
        # (n_batch, seq_len + 1, 512), (n_batch, seq_len, 512)
        text_feats, video_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        shuffle_cls_feats = self.pooler(x)
        
        # metrics
        # itm

        # (n_batch, 2)
        itm_logits = self.itm_score(cls_feats)

        shuffle_logits = self.itm_score(shuffle_cls_feats)
        
        # shuffle
        # (n_batch, seq_len, seq_len)
        shuffle_logits = self.shuffle_score(video_feats)
        
        if return_logist : 
            return itm_logits, shuffle_logits
        
        # itm_labels.sum() == 32
        itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
        shuffle_loss = F.cross_entropy(shuffle_logits.view(-1, self.seq_len), s_labels.long().view(-1), ignore_index = -100)

        return itm_loss, shuffle_loss 

    def embed_texts(self, texts, text_masks):
        batch = texts.shape[0]
        
        texts = texts.reshape(-1, texts.shape[-1])
        text_masks = torch.cat([torch.ones(batch, 1).cuda(), text_masks], dim = 1)

        with torch.no_grad():
            _, text_embeds = text_embedder(texts)

        text_embeds = text_embeds.reshape(batch, -1, text_embeds.shape[-1]).float()
        
        cls_embeds = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(text_embeds.shape[0], 1, 1)
        
        text_embeds = torch.cat([cls_embeds, text_embeds], dim = 1)
        
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks, dtype = torch.long))
        
        text_embeds = text_embeds + self.text_pos_emb(torch.arange(text_embeds.shape[1]).cuda())

        return text_embeds, text_masks


    def inference(self, shots, text_embeds, text_masks, itm_masks, image_token_type_idx = 1):
        
        # itm task
        frame_embeds = shots

        # use key frame
        video_embeds = self.image2video(frame_embeds, text_embeds)
        
        video_embeds = video_embeds + self.token_type_embeddings(torch.full_like(itm_masks, image_token_type_idx, dtype = torch.long))
        
        video_embeds = video_embeds + self.shot_pos_emb(torch.arange(video_embeds.shape[1]).cuda())
        
        co_embeds = torch.cat([text_embeds, video_embeds], dim=1)
        co_masks = torch.cat([text_masks, itm_masks], dim=1)
        
        co_masks = ~co_masks.bool()
    
        x = self.transformer(co_embeds.transpose(0, 1), src_key_padding_mask = co_masks).transpose(0,1)

        cls_feats = self.pooler(x)
        
        itm_logits = self.itm_score(cls_feats)

        return itm_logits 
        