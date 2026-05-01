import ast
import pandas as pd

df = pd.read_parquet('data/factorywave/data/episode.parquet')
meta = df['episode_metadata'].apply(ast.literal_eval)
robot = meta.apply(lambda m: m.get('robot_model'))
cond = meta.apply(lambda m: m.get('condition'))

print('all kuka episodes by condition:')
print(cond[robot == 'kuka'].value_counts(dropna=False))

raw = df['episode_metadata']
mask_230 = raw.str.contains(r'\[0,\s*0,\s*230\]', regex=True, na=False)
mask_160 = raw.str.contains(r'\[0,\s*0,\s*160\]', regex=True, na=False)
print()
print('episodes mentioning [0,0,230] anywhere:', int(mask_230.sum()))
print('episodes mentioning [0,0,160] anywhere:', int(mask_160.sum()))

if mask_230.any():
    print()
    print('[0,0,230] by robot x condition:')
    print(pd.crosstab(robot[mask_230], cond[mask_230]))
if mask_160.any():
    print()
    print('[0,0,160] by robot x condition:')
    print(pd.crosstab(robot[mask_160], cond[mask_160]))

toff = meta.apply(lambda m: m.get('tcp_offset_configured'))
cog = meta.apply(lambda m: m.get('payload_cog_configured'))

print()
print('kuka condition x tcp_offset_configured:')
print(pd.crosstab(cond[robot == 'kuka'], toff[robot == 'kuka'], dropna=False))

print()
print('kuka condition x payload_cog_configured:')
print(pd.crosstab(cond[robot == 'kuka'], cog[robot == 'kuka'], dropna=False))

print()
print('kuka condition x (tcp_offset, cog) jointly:')
joint = list(zip(toff[robot == 'kuka'], cog[robot == 'kuka']))
from collections import Counter
print(Counter(zip(cond[robot == 'kuka'], joint)).most_common())
