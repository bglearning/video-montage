import json
import os
import random
import sys

from pathlib import Path
from typing import List, Union

from moviepy.editor import VideoFileClip, concatenate_videoclips


def concat_and_save(
    clips: List[Union[str, os.PathLike]],
    output_path: Union[str, os.PathLike]
) -> None:
    clips = [VideoFileClip(str(clip)) for clip in clips]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path)


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
    if not len(sys.argv) in (2, 3):
        raise ValueError("Invocation: <script> <shots.jsonl> [<index>]")
    
    shots_file = sys.argv[1]
    instances = []
    with open(shots_file) as f:
        for l in f.readlines():
            instances.append(json.loads(l))
    index = random.randint(0, len(instances) - 1)
    if len(sys.argv) == 3:
        index = int(sys.argv[2])
    print('###\n' + instances[index]['caption'] + '\n\n')
    shots = instances[index]['shots']
    paths = shot_ids_to_paths(shots)
    concat_and_save(paths, shots_file.replace(".jsonl", f"-{index}.mp4"))


if __name__ == '__main__':
    main()
