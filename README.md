# <p align=center>`RATV`</p>

From the paper *Shot Retrieval and Assembly with Text Script for Video Montage Generation* in PyTorch.

Code derived from the supplementary material section [here](https://openreview.net/forum?id=3owqfawaLv).

## Datasets

Original (paper's) VSPD dataset at https://github.com/RATVDemo/RATV

## Setup / Installation

Assuming the `RATV/` dir as root...

### Env

```
pip install -r requirements.txt
```

### Data Download

For VSPD:

```
wget -O shots.tar "https://drive.google.com/uc?export=download&id=1wL3tgbDmzHL0arkBP_9SsLrOZQuiP1HL&confirm=t"
tar -xvf shots.tar
wget -O train_captions.jsonl "https://drive.google.com/uc?export=download&id=1vFW_z8WgMYgzw_evhFKs3QyucJ6DwKzT"
wget -O test_captions.jsonl "https://drive.google.com/uc?export=download&id=1s8TdVW6gGGlQYKBYDsyUqr7zkNHjVp94"

```

For tiktok, assuming the bucket `tiktok_montage` is accessible from the instance:
```
gsutil -m cp -r "gs://tiktok_montage/scenes" .
gsutil -m cp "gs://tiktok_montage/tiktok_montage_gt_4w_9k.json" .
```

### Data-Embedding

We need to embed the videos before training and all.

For this, run `embed_videos.py`


## Training

With `train.py`. For instance:

```
export WANDB_API_KEY="..."
export ENCODER_TYPE="blip2"
python ~/RATV/train.py --data_dir ~/RATV/Embeddings_blip2_proj_f5/ --json_file ~/RATV/data/train_captions.jsonl --embed_dim 256 --hidden_size 256 --learning_rate 1e-4 --loss_weight 0.5 --batch_size 128 --wandb_name "RATV-Full" --epochs 200
```

## Generation

With `generate.py`. For instance:

```
export ENCODER_TYPE="blip2"
python ~/RATV/generate.py \
--data_dir ~/RATV/Embeddings_blip2_f5/ \
--data_dir_proj ~/RATV/Embeddings_blip2_proj_f5/ \
--json_file ~/RATV/data/test_captions.jsonl \
--transformer_path ~/RATV/checkpoints-glad-bee/transformer_pretrained_40.pt \
--threshold 0.5 \
--weight 0.75 \
--hidden_size 768 \
--sim_func default \
--allow_multi_embed \
--output_file ~/RATV/test_captions_generated.jsonl
```

## Evaluation

With `evaluate.py`. For instance:

```
export ENCODER_TYPE="blip2"
python ~/RATV/evaluate.py \
--fea_root ~/RATV/Embeddings_blip2_proj_f5/ \ 
--target_file ~/RATV/data/test_captions.jsonl \
--generated_file ~/RATV/test_captions_generated.jsonl
```

Note: for consistency, we evaluate all outputs with BLIP2 embeddings.