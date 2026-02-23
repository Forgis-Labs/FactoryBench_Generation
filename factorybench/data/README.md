# Data Normalization Tools

This directory contains scripts to normalize various dataset formats into standardized schemas for FactoryBench.

## Dataset Installers & Normalizer (AURSAD / CNC)

For AURSAD and CNC datasets, use the installer and normalizer scripts in `src/data`:

```bash
# AURSAD: download + export experiment_*.csv
python -m src.data.data_installation.install_aursad --max-timestamps 100000

# CNC: download + extract + add timestamp_ms
python -m src.data.data_installation.install_cnc --setup
python -m src.data.data_installation.install_cnc

# Normalize all experiment_*.csv to JSON
python -m src.data.data_normalization.mapped_dataset_normalizer \
    --dataset aursad --input datasets/open_datasets/aursad --output datasets/normalized_episodes
```

## Tools Available

### 1. CWRU Bearing Dataset Converter (`cwru_converter.py`)

Converts CWRU bearing fault diagnosis .mat files to more accessible formats (CSV and/or HDF5).

**Location:** `factorybench/data/cwru_converter.py`

**Source Data:** `datasets/open_datasets/CWRU/cwru/`

**Usage:**

```bash
python -m factorybench.data.cwru_converter \
    --input-dir datasets/open_datasets/CWRU/cwru \
    --output-dir datasets/open_datasets/CWRU/converted \
    --format both  # Options: csv, hdf5, both
```

**Output:**

- `csv/`: Tabular CSV files with metadata
- `hdf5/`: HDF5 files with hierarchical structure
- `csv/metadata.json` and `hdf5/metadata.json`: Dataset documentation

**Features:**

- Automatically parses fault types and load conditions from filenames
- Handles variable-length signals across files
- Includes comprehensive metadata about each file

### 2. UR3e Dataset Normalizer (`ur3e_normalizer.py`)

Normalizes UR3+CobotOps Excel datasets to UR3e schema JSON format as specified in `datasets/ur3e_schema.md`.

**Location:** `factorybench/data/ur3e_normalizer.py`

**Source Data:** `datasets/open_datasets/ur3+cobotops/dataset_02052023.xlsx`

**Usage:**

```bash
python -m factorybench.data.ur3e_normalizer \
    --input datasets/open_datasets/ur3+cobotops/dataset_02052023.xlsx \
    --output datasets/open_datasets/ur3+cobotops/normalized \
    --episode-id ur3e_episode_001
```

**Output:**

- `ur3e_episode_001.json`: Array of normalized time-series records
- `ur3e_episode_001_metadata.json`: Episode metadata and schema information

**Features:**

- Maps Excel columns to UR3e schema
- Converts timestamps to milliseconds since episode start
- Replaces missing schema columns with `null`
- Generates comprehensive metadata about available/missing data
- Processes 7400+ samples in ~1-2 minutes

**Mapping Example:**

| Excel Column           | UR3e Schema Field       | Unit  | Available |
| ---------------------- | ----------------------- | ----- | --------- |
| `Current_J0`           | `effort_current_0`      | A     | ✓         |
| `Speed_J0`             | `feedback_speed_0`      | rad/s | ✓         |
| `Temperature_T0`       | `joint_temp_0`          | °C    | ✓         |
| `Robot_ProtectiveStop` | `protective_stop_state` | {0,1} | ✓         |
| _N/A_                  | `feedback_pos_*`        | rad   | ✗ (null)  |
| _N/A_                  | `setpoint_pos_*`        | rad   | ✗ (null)  |
| _N/A_                  | `true_force_*`          | N/Nm  | ✗ (null)  |

## Schema Information

### UR3e Schema

See `datasets/ur3e_schema.md` for complete specification including:

- Time indexing conventions
- INTENT signals (joint commands, TCP commands, gripper)
- CONTEXT signals (temperatures, voltage, safety)
- OUTCOME signals (joint feedback, forces, vibration, acoustic emission)

### CWRU Metadata

The converter automatically extracts:

- **Fault Types:** Normal, B007, B014, B021, IR007, IR014, IR021, OR0076, OR0146, OR0216
- **Load Conditions:** 0 hp, 1 hp, 2 hp, 3 hp
- **Signal Types:** Motor temperature, speed, current, etc.

## Dependencies

All required packages are listed in `pyproject.toml`:

- `pandas` - Data manipulation
- `scipy` - .mat file reading
- `numpy` - Numerical operations
- `openpyxl` - Excel file reading (auto-installed as pandas dependency)

## Examples

### Convert CWRU to CSV:

```bash
python -m factorybench.data.cwru_converter \
    --input-dir datasets/open_datasets/CWRU/cwru \
    --output-dir datasets/open_datasets/CWRU/converted_csv \
    --format csv -v
```

### Normalize UR3e with custom episode ID:

```bash
python -m factorybench.data.ur3e_normalizer \
    --input datasets/open_datasets/ur3+cobotops/dataset_02052023.xlsx \
    --output datasets/open_datasets/ur3+cobotops/normalized \
    --episode-id ur3e_cobotops_20221026 \
    --no-metadata  # Skip metadata file
```

## Output Verification

**UR3e Normalized Output Sample:**

```json
{
  "timestamp_ms": 0,
  "joint_temp_0": 27.875,
  "effort_current_0": 0.109628,
  "feedback_speed_0": 0.295565,
  "protective_stop_state": 0.0,
  "setpoint_pos_0": null,
  "vibration_0": null,
  ...
}
```

All 7409 samples processed with proper timestamp normalization (relative to first sample at t=0ms).

## Troubleshooting

**Issue:** "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install pandas scipy numpy openpyxl
```

**Issue:** Excel file with many sheets

**Solution:** The normalizer processes the active (first) sheet by default. To specify a different sheet, modify the `normalize_dataset()` function call in the script.

**Issue:** Large datasets causing memory errors

**Solution:** Both scripts process data incrementally and should handle datasets with 10k+ samples. For very large files (>100MB), consider processing in chunks by modifying the respective converter functions.
