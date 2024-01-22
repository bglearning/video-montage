import gc
import json

from pathlib import Path

from clip_video_encode import clip_video_encode

EMBEDDING_DIR_BASE = Path.home() / "Embeddings_blip2_embeds_f20_k8/"
Path(EMBEDDING_DIR_BASE).mkdir(exist_ok=True)

SCENE_DIR = str(Path.home() / Path("scenes/"))

with open("tiktok_montage_gt_4w_5k.json") as f:
    tm_train = json.load(f)

vids_to_encode = []
vid_ids = []

for tm_ in tm_train:
    # Get '/<scene_id>/<num>.mp4' from '.../<scene_id>/<num>.mp4'
    scene_ids = [s['path'].split('scenes')[1] for s in tm_['scenes'] if 'mp4' in s['path']]

    if len(scene_ids) == 0:
        print(f"No scene id for {tm_['id']}")

    # "/<scene_id>/<num>.mp4" -> <scene_id>_<num>
    vid_ids.extend([si[1:-4].replace("/", "_") for si in scene_ids])
    vids_to_encode.extend([SCENE_DIR + f'{v}' for v in scene_ids])

print(len(vids_to_encode))
print(len(vid_ids))

with open(EMBEDDING_DIR_BASE / "mapping_list_4k.json") as f:
    mappings_4k = json.load(f)

print(len(mappings_4k))
full_mapping = mappings_4k + vid_ids
print(len(vid_ids))
print(len(full_mapping))

with open(EMBEDDING_DIR_BASE / "mapping_gt_t2_9k.json", "w") as f:
    json.dump(full_mapping, f)

EMBEDDING_DIR_BASE = str(EMBEDDING_DIR_BASE)
print(EMBEDDING_DIR_BASE)

BATCH_SIZE = 200

del tm_train
del vid_ids
gc.collect()

for i in range(0, len(vids_to_encode) // BATCH_SIZE + 1):
    start = i * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(vids_to_encode))
    vids = vids_to_encode[start: end]
    print(f'Processing {start=}, {end=}')

    # Offset by the 4k's mapping already done
    output_key_start = len(mappings_4k) + start

    clip_video_encode(vids,
        dest=EMBEDDING_DIR_BASE,
        output_format="files",
        take_every_nth=20,
        encoder_type="blip2_both",
        output_key_start=output_key_start,
        frame_workers=4,
        n_final_frames=8,
    )
    gc.collect()
