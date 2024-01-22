import json

from pathlib import Path

base_folder = Path("Embeddings_blip2_embeds_f20_k8")

with open(base_folder / "mapping_gt_t2_9k.json") as f:
    ids = json.load(f)

print('Num of scene ids: ', len(ids))
print('Num of unique scene ids: ', len(set(ids)))

all_embeds = list(base_folder.glob("*.npy"))
ids_with_embeds = set(int(p.stem) for p in all_embeds)
print('Num of embeds: ', len(all_embeds))
print('Num of ids with embeds: ', len(ids_with_embeds))

mapped_gt = {}

for i, full_id in enumerate(ids):
    if i in ids_with_embeds:
        # Same ids get mapped to the embedding of the last occurrence
        mapped_gt[full_id] = i

print('Final num of ids with embeds (deduplicated): ', len(mapped_gt))
with open(base_folder / "mapping_gt_t2_9k_selected.json", "w") as f:
    json.dump(mapped_gt, f)

# Final dataset creation
# From the 9k, only select the instances that have embeddings
# And convert it into a more flattened format

with open("tiktok_montage_gt_4w_9k.json") as f:
    tm_all = json.load(f)
print("Num of all instances: ", len(tm_all))

with open("industry_labels.json") as f:
    label_str = f.read().replace("\\'", "'")
    industry_labels = json.loads(label_str)

final_dataset = []

unknown_industry_labels = []

for tm_ in tm_all:
    scene_ids  = [s['path'].split('scenes')[1] for s in tm_['scenes'] if 'mp4' in s['path']]
    meta = tm_['source_meta']
    if len(scene_ids) == 0:
        continue
    industry = industry_labels.get(meta.get('industry_key', ''), '')
    if industry == '':
        unknown_industry_labels.append(tm_.get('industry_key', ''))
    scene_mapping_ids = [s[1:-4].replace("/", "_") for s in scene_ids]

    # If all scenes have an embedding, accept the instance
    if all([sm in mapped_gt for sm in scene_mapping_ids]):
        # Deduplicate and sort by final _<num> 
        scene_mapping_ids_sorted = sorted(list(set(scene_mapping_ids)), key=lambda s: int(s.split('_')[1]))
        final_dataset.append({'id': tm_['id'], 'texts': meta['all_titles'], 'industry': industry, 'shots': scene_mapping_ids_sorted})
    
print("Final dataset size: ", len(final_dataset))

# Check for duplicates
import pandas as pd
print("n of unknown industries: ", len(unknown_industry_labels))
final_df = pd.DataFrame(final_dataset)
final_df['t'] = final_df['texts'].map(lambda x: ' '.join(x))
final_df['s'] = final_df['shots'].map(lambda x: ' '.join(x))
print("Duplicates: ", final_df[['t', 's']].duplicated().sum())

with open("tiktok_montage_gt_4w_9k_emb.json", "w") as f:
    json.dump(final_dataset, f)
