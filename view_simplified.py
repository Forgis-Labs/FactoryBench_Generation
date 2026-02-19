import json

data = json.load(open('datasets/questions/dummy/test_simplified.json'))

# Group by type
q_by_type = {}
for q in data:
    idx = q['question']['type']
    if idx not in q_by_type:
        q_by_type[idx] = q

type_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5', 6: 'Q6'}

for t in sorted(q_by_type.keys()):
    q = q_by_type[t]
    q_name = type_names.get(t, f'Type{t}')
    reasoning = q['reasoning']
    
    print(f"\n{'='*80}")
    print(f"{q_name}")
    print(f"{'='*80}")
    print(reasoning)
