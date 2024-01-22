import json
import os

from collections import defaultdict
from typing import Dict, List, Union

from datasets.common import TextVidDataset


def load_captions(caption_file: Union[str, os.PathLike]) -> Dict:
    all_captions = defaultdict(list)
    with open(caption_file) as f:
        for l in f.readlines():
            # {"/some/path/<id>/<num>.mp4" : [list]}
            # -> {"<id>_<num>": [list]}
            captions = json.loads(l)
            # Actually should only contain one item
            for k, v in captions.items():
                vid_id, clip_id = k.split("/")[-2:]
                clip_id = clip_id[:-4]
                all_captions[f"{vid_id}_{clip_id}"] += v
    return all_captions


class TiktokDataset(TextVidDataset):
    
    def get_text(self, index: int) -> List[str]:
        item = self.instances[index]
        text = '.'.join(item['texts'])
        if item['industry'] != '':
            text = item['industry'] + ': ' + text
        sentences = text.strip().split('.')
        sentences = [self.punctuate(s) for s in sentences if s != '']
        return sentences


class TiktokDatasetCaps(TiktokDataset):
    def get_text(self, index: int) -> List[str]:
        item = self.instances[index]
        sentences = []
        for scene in item['shots']:
            # Get the middle caption
            ins_captions = self.captions.get(scene, [])
            if len(ins_captions) > 0:
                mid = len(ins_captions) // 2
                sentences.append(ins_captions[mid])
        # If there aren't any captions, return normal text
        if len(sentences) == 0:
            return super().get_text(index)
        return sentences
        