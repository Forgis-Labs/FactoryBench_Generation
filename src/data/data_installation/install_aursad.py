from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

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


def dataset_to_dataframe(ds: h5py.Dataset, prefix: str) -> pd.DataFrame:
    data = ds[...]

    if np.isscalar(data):
        return pd.DataFrame({f"{prefix}__value": [data]})

    if data.dtype.fields is not None:
        df = pd.DataFrame(data)
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

def main() -> None:
    ap = argparse.ArgumentParser(description="Download AURSAD and export a combined CSV")
    repo_root = Path(__file__).resolve().parents[3]
    default_out_dir = repo_root / "datasets" / "open_datasets" / "aursad"
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out_dir),
        help="Where to store dataset files",
    )
    ap.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional CSV path (default: <out-dir>/AURSAD.csv)",
    )
    ap.add_argument("--max-rows", type=int, default=None, help="Optional row limit per dataset")
    ap.add_argument("--skip-md5", action="store_true", help="Skip checksum verification")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    h5_path = out_dir / "AURSAD.h5"
    csv_path = Path(args.csv_path).resolve() if args.csv_path else out_dir / "aursad.csv"

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

    metadata: Dict[str, Dict[str, Any]] = {}
    frames: List[pd.DataFrame] = []

    with h5py.File(h5_path, "r") as f:
        datasets = collect_datasets(f)
        for ds in tqdm(datasets, desc="Converting datasets", unit="dataset"):
            h5_dataset_path = ds.name
            prefix = h5_dataset_path.strip("/").replace("/", "__")
            df = dataset_to_dataframe(ds, prefix)

            if args.max_rows is not None:
                df = df.head(args.max_rows)

            frames.append(df)
            metadata[h5_dataset_path] = {
                "prefix": prefix,
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
            }

    if frames:
        combined = pd.concat(frames, axis=1)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(csv_path, index=False)

    index_path = csv_path.with_suffix(".index.json")
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if h5_path.exists():
        h5_path.unlink()

    print(f"OK: {h5_path}")
    print(f"CSV: {csv_path}")
    print(f"Index: {index_path}")

if __name__ == "__main__":
    main()
