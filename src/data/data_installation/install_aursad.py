from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.data._decimation import decimate_dataframe

# Zenodo "latest" record (v1.1) direct file links:
AURSAD_H5_URL = "https://zenodo.org/records/4559556/files/AURSAD.h5?download=1"
AURSAD_H5_MD5 = "08e4706cf15144761a12cb86bd071d72"

def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support if partial file exists
    headers = {}
    mode = "wb"
    existing = out_path.stat().st_size if out_path.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()

        total = r.headers.get("Content-Length")
        total = int(total) + existing if total is not None else None

        desc = f"Downloading {out_path.name}"
        with tqdm(total=total, initial=existing, unit="B", unit_scale=True, desc=desc) as pbar:
            with open(out_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def dataset_to_dataframe(
    ds: h5py.Dataset,
    prefix: str,
    target_rows: Optional[int],
) -> pd.DataFrame:
    if target_rows is not None and ds.shape and len(ds.shape) > 0:
        data = ds[:target_rows]
    else:
        data = ds[...]

    if np.isscalar(data):
        if target_rows is not None:
            return pd.DataFrame({f"{prefix}__value": [data] * target_rows})
        return pd.DataFrame({f"{prefix}__value": [data]})

    if data.dtype.fields is not None:
        df = pd.DataFrame(data)
        if target_rows is not None:
            df = df.iloc[:target_rows]
        return df.add_prefix(f"{prefix}__")

    if data.ndim == 1:
        return pd.DataFrame({f"{prefix}__value": data})

    if data.ndim == 2:
        cols = [f"{prefix}__feature_{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols)

    flat = data.reshape(data.shape[0], -1)
    cols = [f"{prefix}__feature_{i}" for i in range(flat.shape[1])]
    return pd.DataFrame(flat, columns=cols)


def collect_datasets(h5_file: h5py.File) -> List[h5py.Dataset]:
    datasets: List[h5py.Dataset] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(obj)

    h5_file.visititems(visitor)
    return datasets


def export_by_experiments(
    data_frame: pd.DataFrame,
    out_dir: Path,
    max_timestamps: Optional[int] = None,
) -> None:
    """
    Export DataFrame organized by experiments based on 'sample_nr' column.
    Each experiment is written as experiment_{i}.csv in the output directory.
    Multiplies any timestamp columns by 1000 to convert to milliseconds.
    """
    if "sample_nr" not in data_frame.columns:
        print("⚠ 'sample_nr' column not found. Exporting as single CSV instead.")
        csv_path = out_dir / "AURSAD.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        data_frame.to_csv(csv_path, index=False)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Multiply timestamp columns by 1000 to convert to milliseconds
    for col in data_frame.columns:
        if "timestamp" in col.lower():
            # Convert to float first if needed, then multiply and convert to int
            data_frame[col] = (data_frame[col] * 1000).astype('int64')
    
    # Group by sample_nr (experiment ID)
    grouped = data_frame.groupby("sample_nr", sort=True)
    
    print(f"\nExporting {len(grouped)} experiments to subfolders...")
    
    for sample_nr, group_df in grouped:
        exp_num = int(sample_nr) if isinstance(sample_nr, (int, np.integer)) else sample_nr
        csv_path = out_dir / f"experiment_{exp_num}.csv"
        group_df.to_csv(csv_path, index=False)
        
        print(f"  ✓ experiment_{exp_num}: {len(group_df)} rows")
    
    print(f"✓ All experiments exported to {out_dir}\n")

def main() -> None:
    ap = argparse.ArgumentParser(description="Download AURSAD and export experiments to separate CSVs")
    repo_root = Path(__file__).resolve().parents[3]
    default_out_dir = repo_root / "data" / "open_datasets" / "aursad"
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out_dir),
        help="Where to store dataset files",
    )
    ap.add_argument(
        "--max-timestamps",
        type=int,
        default=None,
        help="Optional limit on number of timestamps (rows) to export",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="(Deprecated) Use --max-timestamps instead",
    )
    ap.add_argument("--skip-md5", action="store_true", help="Skip checksum verification")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    h5_path = out_dir / "AURSAD.h5"
    index_path = out_dir / "aursad.index.json"

    if h5_path.exists():
        print(f"Found existing dataset: {h5_path}")
    else:
        download(AURSAD_H5_URL, h5_path)

    if not args.skip_md5:
        got = md5sum(h5_path)
        if got.lower() != AURSAD_H5_MD5.lower():
            raise RuntimeError(
                f"MD5 mismatch for {h5_path}\n"
                f"Expected: {AURSAD_H5_MD5}\n"
                f"Got:      {got}\n"
                "Delete the file and rerun to re-download."
            )

    if args.max_timestamps is not None and args.max_timestamps < 1:
        raise ValueError("--max-timestamps must be >= 1")

    max_timestamps = args.max_timestamps
    if max_timestamps is None and args.max_rows is not None:
        max_timestamps = args.max_rows
    
    # Read 10x more rows than requested so decimation by 10 yields the target count
    hdf5_read_limit = None
    if max_timestamps is not None:
        hdf5_read_limit = max_timestamps * 10

    metadata: Dict[str, Dict[str, Any]] = {}

    data_frame: Optional[pd.DataFrame] = None
    with pd.HDFStore(h5_path, mode="r") as store:
        if "/complete_data" in store.keys():
            try:
                result = store["complete_data"]
                if not isinstance(result, pd.DataFrame):
                    result = result.to_frame()
                data_frame = result
                
                if data_frame is not None and hdf5_read_limit is not None:
                    data_frame = data_frame.head(hdf5_read_limit)

                if data_frame is not None:
                    metadata["/complete_data"] = {
                        "rows_exported": int(len(data_frame)),
                        "columns": list(data_frame.columns),
                        "dtypes": {col: str(dtype) for col, dtype in data_frame.dtypes.items()},
                    }
            except Exception:
                data_frame = None

    if data_frame is None:
        frames: List[pd.DataFrame] = []
        with h5py.File(h5_path, "r") as f:
            datasets = collect_datasets(f)
            lengths: List[int] = []
            for ds in datasets:
                if ds.shape and len(ds.shape) > 0 and ds.shape[0] > 1:
                    lengths.append(ds.shape[0])

            base_len = min(lengths) if lengths else 1
            target_rows = base_len if hdf5_read_limit is None else min(hdf5_read_limit, base_len)

            for ds in tqdm(datasets, desc="Converting datasets", unit="dataset"):
                h5_dataset_path = ds.name
                prefix = h5_dataset_path.strip("/").replace("/", "__")
                df = dataset_to_dataframe(ds, prefix, target_rows)

                frames.append(df)
                metadata[h5_dataset_path] = {
                    "prefix": prefix,
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtype),
                    "rows_exported": target_rows,
                }

        if frames:
            data_frame = pd.concat(frames, axis=1)

    if data_frame is not None:
        # Anti-aliased downsample (10x) per experiment to avoid cross-experiment filter artifacts
        _NON_CONTINUOUS = {"sample_nr", "label"}
        _continuous = set(data_frame.columns) - _NON_CONTINUOUS
        if "sample_nr" in data_frame.columns:
            decimated_groups = []
            for _, group_df in data_frame.groupby("sample_nr", sort=True):
                decimated_groups.append(
                    decimate_dataframe(group_df, q=10, continuous_cols=_continuous)
                )
            data_frame = pd.concat(decimated_groups, ignore_index=True)
        else:
            data_frame = decimate_dataframe(data_frame, q=10, continuous_cols=_continuous)

        # Apply max_timestamps limit after downsampling
        if max_timestamps is not None and len(data_frame) > max_timestamps:
            data_frame = data_frame.head(max_timestamps)

        # Export by experiments into experiment_{i}/ subfolders
        export_by_experiments(data_frame, out_dir, max_timestamps)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"OK: {h5_path}")
    print(f"Index: {index_path}")

if __name__ == "__main__":
    main()
