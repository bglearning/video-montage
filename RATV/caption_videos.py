import json
import sys

import numpy as np
import torch

from pathlib import Path
from typing import List

from PIL import Image

from lavis.models import load_model_and_preprocess
from video2numpy.frame_reader import FrameReader


def read_video_frames(vids):
    take_every_5 = 30
    resize_size = 300
    batch_size = 8 # output shape will be (n, batch_size, height, width, 3)
    n_final_frames = 5

    reader = FrameReader(vids,
                        take_every_nth=take_every_5,
                        resize_size=resize_size,
                        workers=4,
                        batch_size=batch_size)
    reader.start_reading()

    sampled_vid_frames = []
    ids = []

    for vid_frames, info_dict in reader:
        # info_dict["dst_name"] - name for saving numpy array
        # info_dict["pad_by"] - how many pad frames were added to final block so n_frames % batch_size == 0
        # do something with vid_frames of shape (n_blocks, 64, 300, 300, 3)
        n_blocks, bs, h, w, c = vid_frames.shape
        vid_frames = vid_frames.reshape(n_blocks * bs, h, w, c)
        sample_idx = np.round(np.linspace(0, len(vid_frames) - 1, n_final_frames)).astype(int)
        sample_idx = np.unique(sample_idx)
        sampled_vid_frames.append(vid_frames[sample_idx])
        ids.append(info_dict['reference'])
    return sampled_vid_frames, ids

def shot_ids_to_paths(shot_ids: List[str]):
    SCENE_DIR = Path.home() / Path("scenes/")
    paths = []
    for shot_id in shot_ids:
        # Reverse changes from embed_videos
        # <shot-id>_<shot-num> => <shot-id>/<shot-num>.mp4
        shot_file = '/'.join(shot_id.split("_")) + '.mp4'
        paths.append(SCENE_DIR / shot_file)
    return paths


def main():

    if len(sys.argv) != 3:
        raise ValueError("Invocation: <script> <data.json> <output.jsonl>")

    data_file = sys.argv[1]
    with open(data_file) as f:
        instances = json.load(f)
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
    )

    out_file = sys.argv[2]
    with open(out_file, 'w') as f:
        for instance in instances:
            vids = [str(p) for p in shot_ids_to_paths(instance['scenes'])]
            sampled_vids_frames, ids = read_video_frames(vids)

            for vid_frames, i in zip(sampled_vids_frames, ids):
                # Skip empty frames
                # Note: means we can't rely on captions and frames being zippable
                captions = []
                for fr in vid_frames:
                    if fr.sum() == 0:
                        continue
                    image = Image.fromarray(fr.astype('uint8'), 'RGB')
                    image = vis_processors["eval"](image).unsqueeze(0).to(device)
                    captions.extend(model.generate({"image": image}))
                f.write(json.dumps({vids[i]: captions}) + "\n")


if __name__ == '__main__':
    main()
