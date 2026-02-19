import json
with open('datasets/questions/dummy/test_q5.json') as f:
    data = json.load(f)

# Find Q5 with numeric answer
for q in data:
    if q['question']['type'] == 5 and isinstance(q['answer'], (int, float)):
        print("Q5 with jerk value found:")
        print(f"Answer: {q['answer']}")
        print(f"Reasoning:\n{q['reasoning']}")
        break
else:
    print("No Q5 with numeric answer found in dataset")
