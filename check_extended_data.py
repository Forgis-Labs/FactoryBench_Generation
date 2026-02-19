import json

data = json.load(open('datasets/normalized_episodes/dummy/ABB_extended.json'))
non_zero = [i for i, s in enumerate(data) if s.get('fault_id', 0) != 0]

print(f'Total samples: {len(data)}')
print(f'Non-zero fault_id samples: {len(non_zero)}')
print(f'First 10 non-zero fault indices: {non_zero[:10]}')
print(f'Their fault_ids: {[data[i]["fault_id"] for i in non_zero[:10]]}')
print(f'\nSample with non-zero fault_id (index {non_zero[0]}):')
print(f'  fault_id: {data[non_zero[0]]["fault_id"]}')
print(f'  timestamp_ms: {data[non_zero[0]]["timestamp_ms"]}')
