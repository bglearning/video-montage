import os

from pathlib import Path
from typing import List, Union

from loguru import logger

from datasets.common import TextVidDataset


class VSPD:
    NUM_VIDS = 19_613

    def __init__(self, root_dir: Union[os.PathLike, str]) -> None:
        self.root_dir = Path(root_dir).expanduser()

        # Sanity check
        num_vids = len(list(self.root_dir.glob("*.mp4")))
        if num_vids != self.NUM_VIDS:
            logger.warning(
                f"Num vids in VSPD directory {self.root_dir}: {num_vids}"
                f"\nBut expected {self.NUM_VIDS}"
            )

    def paths(self):
        # The paths are just integers for each vid
        paths = []
        ids = []
        for v in range(0, self.NUM_VIDS):
            # The ids are just the integer indices themselves
            ids.append(v)
            # The path is directly based on the integer index
            paths.append(self.root_dir / f'{v}.mp4')

        return paths, ids


class VSPDDataset(TextVidDataset):

    def get_text(self, index: int) -> List[str]:
        item = self.instances[index]
        text = item['caption']
        sentences = text.strip().split('.')
        sentences = [s + '.' for s in sentences if s != '']
        return sentences
        