# %%
import json
import pandas as pd
import sys

score_jsonl_file = sys.argv[1]

test_cases = []
with open(score_jsonl_file) as f:
    for l in f.readlines():
        test_cases.append(json.loads(l))
# %%
score_details = []
for test_case in test_cases:
    for i, score in enumerate(test_case['scores']):
        score_details.append((i + 1, score['tot'], score['sc_m'], score['sc_s']))
        # score_details.append((i + 1, score['tot']))

df = pd.DataFrame(score_details, columns=['rank', 'sc_tot', 'sc_m', 'sc_s'])
# df = pd.DataFrame(score_details, columns=['rank', 'sc_tot'])
df.head()
# %%
print(df.describe().T)
# %%
print(df.groupby('rank').mean().T)
# %%

