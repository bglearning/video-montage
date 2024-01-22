import argparse
import gc
import os

from pathlib import Path
from typing import List, Union

from loguru import logger

from clip_video_encode import clip_video_encode

from datasets import VSPD


def embed_vids(vids_to_encode: List[Union[str, os.PathLike]],
               out_dir: str,
               encoder_type: str = "blip2_both",
               batch_size: int = 100):
    """Embed the videos in batches
    """
    # clip_video_encode expects both vids and dest to be strings
    # So we explicitly cast it as so
    vids_to_encode = [str(vid) for vid in vids_to_encode]

    for i in range(0, len(vids_to_encode) // batch_size + 1):
        start = i * batch_size
        end = min(start + batch_size, len(vids_to_encode))
        vids = vids_to_encode[start: end]
        logger.info(f'Processing {start=}, {end=}')

        clip_video_encode(vids,
            dest=out_dir,
            output_format="files",
            take_every_nth=20,
            encoder_type=encoder_type,
            output_key_start=start,
            frame_workers=4,
            n_final_frames=8,
        )
        # Garbage Collect to try to avoid memory leaks
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Embed the raw video files and dump them')
    parser.add_argument('--dataset', required=True, type=str.lower, choices=["vspd", "tiktok-9k"],
                        help='The dataset to be embedded')
    parser.add_argument('--data-dir', required=True, type=str, help='The root directory of the dataset')
    parser.add_argument('--out-dir', required=True, type=str, help='The output directory')
    parser.add_argument('--encoder-type', type=str, default="blip2_both", help='The Encoder type to use')
    parser.add_argument('--batch-size', type=int, default=100, help='Num of vids to embed in a batch')

    args = parser.parse_args()

    logger.info(args)

    # Create the output directory
    EMBEDDING_DIR_BASE = Path(args.out_dir).expanduser()
    EMBEDDING_DIR_BASE.mkdir(exist_ok=True)

    if args.dataset == 'vspd':
        dataset = VSPD(args.data_dir)
    elif args.dataset == 'tiktok-9k':
        ... 
    
    paths, ids = dataset.paths()

    embed_vids(
        vids_to_encode=paths,
        out_dir=args.out_dir,
        encoder_type=args.encoder_type,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
