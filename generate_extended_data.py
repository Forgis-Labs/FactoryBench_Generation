"""
Generate extended dummy data with 100 episodes and fault_id field.

This script:
1. Loads the original dummy data (single episode)
2. Generates 100 episodes by duplicating and randomly perturbing the data
3. Adds a fault_id field (0-5) to each row
   - Most rows have fault_id=0
   - Occasionally (5-10% chance), inject other fault IDs randomly
4. Saves to a new file
"""

import json
import random
from pathlib import Path

# Load original data
input_file = Path("datasets/normalized_episodes/dummy/ABB.json")
output_file = Path("datasets/normalized_episodes/dummy/ABB_extended.json")

print(f"Loading original data from {input_file}...")
with open(input_file) as f:
    original_episode = json.load(f)

print(f"Original episode has {len(original_episode)} samples")

# Generate 100 episodes
all_episodes = []
fault_injection_rate = 0.075  # 7.5% chance of non-zero fault

for episode_num in range(1, 101):
    print(f"Generating episode {episode_num}/100...", end="\r")
    
    episode = []
    for sample_idx, original_sample in enumerate(original_episode):
        # Copy the sample
        sample = original_sample.copy()
        
        # Add fault_id
        # Most of the time use 0, occasionally use 1-5
        if random.random() < fault_injection_rate:
            # Inject a fault ID (1-5)
            sample["fault_id"] = random.randint(1, 5)
        else:
            sample["fault_id"] = 0
        
        # Optional: Add small random perturbations to make episodes slightly different
        # This makes the generated data more realistic
        perturbation_factor = 0.001 * episode_num * random.random()
        
        # Perturb numeric fields slightly (except timestamp)
        for key, value in sample.items():
            if key != "timestamp_ms" and isinstance(value, (int, float)) and key != "fault_id":
                # Add small noise proportional to episode number
                sample[key] = value + (value * perturbation_factor * random.uniform(-1, 1))
        
        episode.append(sample)
    
    all_episodes.extend(episode)

print(f"Generated 100 episodes with {len(all_episodes)} total samples")

# Write output
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(all_episodes, f, indent=2)

print(f"Wrote extended data to {output_file}")

# Print statistics
fault_counts = {}
for sample in all_episodes:
    fault_id = sample.get("fault_id", 0)
    fault_counts[fault_id] = fault_counts.get(fault_id, 0) + 1

print("\nFault ID distribution:")
for fault_id in sorted(fault_counts.keys()):
    count = fault_counts[fault_id]
    percentage = 100.0 * count / len(all_episodes)
    print(f"  fault_id={fault_id}: {count} samples ({percentage:.2f}%)")

print(f"\nTotal samples: {len(all_episodes)}")
print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
