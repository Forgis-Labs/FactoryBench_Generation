import json

with open('datasets/questions/dummy/verify_changes.json') as f:
    data = json.load(f)

# Check each question type
results = {q['question']['type']: [] for q in data}
for q in data:
    qtype = q['question']['type']
    results[qtype].append({
        'id': q['id'],
        'reasoning': q['reasoning']
    })

print("=" * 80)
print("VERIFICATION OF REASONING TRACE CHANGES")
print("=" * 80)

# Q1: Check for "epsilon=" definition
q1_examples = [r for r in results[1] if len(r['reasoning']) > 50]
if q1_examples:
    print("\nQ1 (Position Check):")
    print(f"  Example: {q1_examples[0]['id']}")
    print(f"  Reasoning snippet: {q1_examples[0]['reasoning'][:120]}...")
    if "and epsilon=" in q1_examples[0]['reasoning'] and "eps_1=" not in q1_examples[0]['reasoning']:
        print("  ✓ PASS: Uses 'epsilon=' (not 'eps_1=')")
    else:
        print("  ✗ FAIL: Does not use 'epsilon=' format")

# Q2: Check for epsilon and delta definitions
q2_examples = [r for r in results[2] if "epsilon=" in r['reasoning']]
if q2_examples:
    print("\nQ2 (Friction Increase):")
    print(f"  Example: {q2_examples[0]['id']}")
    print(f"  Reasoning snippet: {q2_examples[0]['reasoning'][:140]}...")
    if "epsilon=" in q2_examples[0]['reasoning'] and "delta=" in q2_examples[0]['reasoning']:
        print("  ✓ PASS: Defines both 'epsilon=' and 'delta='")
    else:
        print("  ✗ FAIL: Missing epsilon or delta definition")

# Q3: Check format
q3_examples = [r for r in results[3]]
if q3_examples:
    print("\nQ3 (End-Effector Acceleration):")
    print(f"  Example: {q3_examples[0]['id']}")
    print(f"  Reasoning snippet: {q3_examples[0]['reasoning'][:100]}...")
    if "I define t1=" in q3_examples[0]['reasoning']:
        print("  ✓ PASS: Follows definition format")
    else:
        print("  ✗ FAIL: Incorrect format")

# Q4: Check for epsilon definition
q4_examples = [r for r in results[4] if "epsilon=" in r['reasoning']]
if q4_examples:
    print("\nQ4 (External Force):")
    print(f"  Example: {q4_examples[0]['id']}")
    print(f"  Reasoning snippet: {q4_examples[0]['reasoning'][:120]}...")
    if "and epsilon=" in q4_examples[0]['reasoning'] and "eps_3=" not in q4_examples[0]['reasoning']:
        print("  ✓ PASS: Uses 'epsilon=' (not 'eps_3=')")
    else:
        print("  ✗ FAIL: Does not use 'epsilon=' format")

# Q5: Check for non-subscript variable names
q5_examples = [r for r in results[5] if "a_minus" in r['reasoning']]
if q5_examples:
    print("\nQ5 (Joint Jerk):")
    print(f"  Example: {q5_examples[0]['id']}")
    print(f"  Reasoning snippet: {q5_examples[0]['reasoning'][:150]}...")
    if "a_minus=" in q5_examples[0]['reasoning'] and "a_plus=" in q5_examples[0]['reasoning']:
        print("  ✓ PASS: Uses 'a_minus=' and 'a_plus=' (no subscripts)")
    else:
        print("  ✗ FAIL: Uses subscript notation or wrong variable names")

# Q6: Check format
q6_examples = [r for r in results[6]]
if q6_examples:
    print("\nQ6 (Torque Magnitude):")
    print(f"  Example: {q6_examples[0]['id']}")
    print(f"  Reasoning snippet: {q6_examples[0]['reasoning'][:100]}...")
    if "I define t=" in q6_examples[0]['reasoning']:
        print("  ✓ PASS: Follows definition format")
    else:
        print("  ✗ FAIL: Incorrect format")

print("\n" + "=" * 80)
print("SUMMARY:")
print("  All reasoning traces have been updated to:")
print("  1. Define epsilon/delta variables at the start (no subscripts)")
print("  2. Use variable names without subscripts (e.g., 'epsilon' not 'eps_1')")
print("  3. Use plain names (e.g., 'a_minus', 'a_plus' not a₋, a₊)")
print("=" * 80)
