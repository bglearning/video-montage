import json
import os

from typing import List, Union

import numpy as np
import torch


# MAPPING_FILE = os.environ.get("MAPPING_FILE", "mapping_gt_t2_9k_selected.json")
MAPPING_FILE = os.environ.get("MAPPING_FILE", None)
SCENE_MAPPINGS = None
# TODO Handle this natively through an argument to FrameReader
if MAPPING_FILE is not None:
    with open(MAPPING_FILE) as f:
        SCENE_MAPPINGS = json.load(f)

class FrameReader:
    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 suffix: str = "",
                 k: int = 8,
                 embed_single: bool = True) -> None:
        self.data_dir = data_dir
        self.suffix = suffix
        self.k = k
        self.embed_single = embed_single
    
    def read(self, shot_strs: List[str]) -> List[torch.Tensor]:

        shots = []
        shots_proj = []
        for shot in shot_strs: 
            if SCENE_MAPPINGS is not None:
                shot_num = SCENE_MAPPINGS[shot]
            else:
                shot_num = shot.split(".")[0]
            frame_path = os.path.join(self.data_dir, f'{shot_num}{self.suffix}.npy')
            
            feat_proj = np.load(frame_path)

            # If each frame embedding is supposed to be single dimension
            # but it's 2 dimensional (so frames is 3 dim), reduce it down
            # E.g. BLIP frames have two dimensional representations
            if self.embed_single and len(feat_proj.shape) == 3:
                feat_proj = feat_proj.mean(axis=1)

            # (n_frames, 32, 768 + 256)
            feat_proj = torch.from_numpy(feat_proj)

            frames_feat, frames_proj = feat_proj[..., :768], feat_proj[..., -256:]

            shots.append(frames_feat)
            shots_proj.append(frames_proj)

        return shots, shots_proj
