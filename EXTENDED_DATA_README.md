# Extended Dummy Data Generation

## Summary

A new extended dummy data file has been generated: `datasets/normalized_episodes/dummy/ABB_extended.json`

## Features

- **Size**: 100 episodes with 1000 total samples (10 samples per episode)
- **New field**: `fault_id` (integer between 0-5 inclusive)
- **Fault distribution**: Approximately 92% fault_id=0, with 1-2% each for fault_ids 1-5

## Data Structure

Each sample in the extended data contains all original fields plus:

```json
{
  "timestamp_ms": 0.0,
  "setpoint_pos_0": 0.3,
  ...all other original fields...
  "fault_id": 0
}
```

## Generation Process

The `generate_extended_data.py` script:

1. Loads the original dummy episode (ABB.json) with 10 samples
2. Creates 100 independent episodes by:
   - Duplicating the original data 100 times
   - Adding small perturbations to numeric fields (realistic variation)
   - Randomly injecting `fault_id` values (7.5% chance of non-zero)
3. Exports all 1000 samples to ABB_extended.json

## Usage

You can now use the extended data for testing:

```bash
python -m src.questions.cli level1 \
  --input datasets/normalized_episodes/dummy/ABB_extended.json \
  --output datasets/questions/dummy/questions_extended.json \
  --n 100
```

## Files Generated

- **generate_extended_data.py**: Script to generate the extended data
- **check_extended_data.py**: Script to verify the extended data
- **ABB_extended.json**: The generated extended data (2.84 MB)

## Fault ID Statistics

```
fault_id=0: 920 samples (92.00%)
fault_id=1:  17 samples (1.70%)
fault_id=2:  13 samples (1.30%)
fault_id=3:  17 samples (1.70%)
fault_id=4:  15 samples (1.50%)
fault_id=5:  18 samples (1.80%)
```
