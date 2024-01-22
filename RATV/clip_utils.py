import os

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_TYPE = os.environ.get('ENCODER_TYPE', 'open_clip')

text_processor, text_embedder = None, None
# text_processor (text: str) -> torch.Tensor <1, seq_len>
# text_embedder (token_ids: torch.Tensor <n, seq_len>) -> <n, embed_dim>

print(f'{ENCODER_TYPE=}')

if ENCODER_TYPE == 'open_clip':
    import open_clip

    OPEN_CLIP_MODEL = 'ViT-B-32'
    clip_model, _, _ = open_clip.create_model_and_transforms(OPEN_CLIP_MODEL,
                                                                pretrained='laion2b_s34b_b79k',
                                                                device=device)
    text_processor = open_clip.get_tokenizer(OPEN_CLIP_MODEL)
    clip_model.eval()
    text_embedder = clip_model.encode_text

elif ENCODER_TYPE == 'openai_clip':
    from clip import clip

    clip_model, _ = clip.load("ViT-B/32", device=device)
    text_processor = partial(clip.tokenize, truncate=True)
    clip_model.eval()
    text_embedder = clip_model.encode_text

elif ENCODER_TYPE == 'blip2':
    from lavis.models import load_model_and_preprocess

    blip_model, vis_preprocessors, text_preprocessors = load_model_and_preprocess("blip2_feature_extractor",
                                                                                  model_type="pretrain",
                                                                                  device=device)
    blip_model.eval()

    text_preprocess = text_preprocessors['eval']
    def text_processor(text: str):
        text = text_preprocess(text)
        text_tokens = blip_model.tokenizer(text,
                                           return_tensors="pt",
                                           padding='max_length',
                                           max_length=77,
                                           truncation=True).to(device)
        return text_tokens.input_ids

    def text_embedder(token_ids):
        # Snippet from `Blip2Qformer.extract_features` with mode='text'
        # Token with id=0 is the padding token (to be masked out)
        attn_mask = (token_ids != 0).to(int).to(device)
        text_output = blip_model.Qformer.bert(
                token_ids,
                attention_mask=attn_mask,
                return_dict=True,
        )
        # (n_batch, n_tokens, embed_dim)
        text_embeds = text_output.last_hidden_state
        # We take full embeddings (768) rather than proj (256)
        # text_features = blip_model.text_proj(text_embeds)
        # text_features = F.normalize(text_features, dim=-1)
        # (n_batch, embed_dim)
        # text_embeds = text_embeds.mean(dim=1)
        text_features = blip_model.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        return text_features[:,0,:], text_embeds.mean(dim=1)
                                                                                