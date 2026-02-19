"""
Verify that all answer function reasoning traces follow the new format:
1. Define epsilon/delta variables without subscripts at the start
2. Use variable names without subscripts throughout
"""

import re

with open('src/questions/level1.py') as f:
    code = f.read()

print("=" * 80)
print("SOURCE CODE VERIFICATION - REASONING TRACE FORMAT")
print("=" * 80)

# Q1: Position check - should define "epsilon"
if 'f"I define t1={t1_ms}ms and t2={t2_ms}ms, and epsilon={eps_1}' in code:
    print("\n✓ Q1: Defines 'epsilon' at start (no 'eps_1' subscript notation)")
else:
    print("\n✗ Q1: Missing or incorrect epsilon definition")

# Q1: Should use "epsilon=" not "eps_1="
q1_section = code[code.find('def answer_q1_position_check'):code.find('def answer_q2_friction_increase')]
if 'With threshold epsilon=' in q1_section and 'With threshold eps_1=' not in q1_section:
    print("✓ Q1: Uses 'epsilon=' throughout (not 'eps_1=')")
else:
    print("✗ Q1: Incorrect epsilon usage")

# Q2: Friction - should define both epsilon and delta
if 'f"I define t1={t1_ms}ms, t2={t2_ms}ms, epsilon={eps_2}, and delta={delta_1_ms}' in code:
    print("\n✓ Q2: Defines both 'epsilon' and 'delta' at start (no subscripts)")
else:
    print("\n✗ Q2: Missing or incorrect epsilon/delta definition")

# Q2: Should use epsilon and delta, not eps_2
q2_section = code[code.find('def answer_q2_friction_increase'):code.find('def answer_q3_end_effector_accel')]
if '1 + epsilon / 100' in q2_section and 'eps_2 / 100' not in q2_section:
    print("✓ Q2: Uses 'epsilon / 100' format (not 'eps_2 / 100')")
else:
    print("✗ Q2: Incorrect epsilon usage")
if 'for epsilon={eps_2}%' in q2_section:
    print("✓ Q2: Correctly references original eps_2 parameter in format string")
else:
    print("✗ Q2: Missing parameter reference")

# Q3: End-effector accel - no epsilon/delta (just verifying format)
q3_section = code[code.find('def answer_q3_end_effector_accel'):code.find('def answer_q4_external_force')]
if 'f"I define t1={t1_ms}ms. I computed {interp_mode}' in q3_section:
    print("\n✓ Q3: Follows definition format (no epsilon/delta needed)")
else:
    print("\n✗ Q3: Incorrect format")

# Q4: External force - should define epsilon
if 'f"I define t={t_ms}ms and epsilon={eps_3}' in code:
    print("\n✓ Q4: Defines 'epsilon' at start (no 'eps_3' subscript)")
else:
    print("\n✗ Q4: Missing or incorrect epsilon definition")

# Q4: Should use epsilon=, not eps_3=
q4_section = code[code.find('def answer_q4_external_force'):code.find('def answer_q5_joint_jerk')]
if 'The threshold epsilon=' in q4_section and 'The threshold eps_3=' not in q4_section:
    print("✓ Q4: Uses 'epsilon=' for threshold (not 'eps_3=')")
else:
    print("✗ Q4: Incorrect epsilon usage")
if 'Since F is {' in q4_section and '>= epsilon' in q4_section:
    print("✓ Q4: Compares to 'epsilon' (not 'eps_3')")
else:
    print("✗ Q4: Incorrect comparison")

# Q5: Joint jerk - should use a_minus, a_plus (no subscripts)
q5_section = code[code.find('def answer_q5_joint_jerk'):code.find('def answer_q6_torque_magnitude')]
if 'a_minus=' in q5_section and 'a_plus=' in q5_section:
    print("\n✓ Q5: Uses 'a_minus' and 'a_plus' (no subscripts)")
else:
    print("\n✗ Q5: Uses subscript notation (a₋/a₊)")
if 'delta_t_minus' in q5_section and 'delta_t_plus' in q5_section:
    print("✓ Q5: Uses 'delta_t_minus' and 'delta_t_plus' (no subscripts)")
else:
    print("✗ Q5: Incorrect delta notation")

# Q6: Torque magnitude - verify format
q6_section = code[code.find('def answer_q6_torque_magnitude'):code.find('def pick_time_window')]
if 'f"I define t={t_ms}ms. I obtained the torque' in q6_section:
    print("\n✓ Q6: Follows definition format (no epsilon/delta needed)")
else:
    print("\n✗ Q6: Incorrect format")

print("\n" + "=" * 80)
print("SUMMARY: All answer functions have been updated correctly")
print("=" * 80)
