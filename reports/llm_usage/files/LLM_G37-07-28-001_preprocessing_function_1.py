"""Preprocessing module for the QUBO-based binary classification project.

Phase 1 of the pipeline:

1. read the input CSV dataset (first row = column headers);
2. separate the target column;
3. drop the features whose percentage of valid values (present AND non-zero)
   is below ``minPercValid``;
4. normalize the remaining features with z-score standardization;
5. write the normalized dataset and a JSON report with the run statistics.

The module is importable (``from qubo_project.preprocessing import fit_normalize``)
and executable from the command line:

    python preprocessing.py \\
        --input dati_credito.csv \\
        --target target \\
        --out-data normalized.csv \\
        --out-json preprocessing_result.json \\
        --min-perc-valid 0.06
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

__all__ = ["fit_normalize", "main"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_parent_dir(path: str) -> Path:
    """Create the parent directory of ``path`` if it does not exist yet.

    Relative paths are resolved against the current working directory, so no
    absolute path is ever hard-coded in the project.
    """
    file_path = Path(path)
    if file_path.parent and str(file_path.parent) not in ("", "."):
        file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _valid_ratio(column: pd.Series) -> float:
    """Fraction of entries that are neither missing nor equal to zero.

    Computed column by column to keep the peak memory footprint low on very
    large datasets (the evaluation dataset has more than 1.5M rows).
    """
    n_rows = len(column)
    if n_rows == 0:
        return 0.0
    valid = int((column.notna() & (column != 0)).sum())
    return valid / n_rows


def _normalize_target(series: pd.Series) -> pd.Series:
    """Return the target as integers when the values are integral (0/1)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and np.array_equal(numeric.to_numpy(), np.round(numeric.to_numpy())):
        return numeric.astype(np.int64)
    return numeric


# --------------------------------------------------------------------------- #
# Mandatory interface
# --------------------------------------------------------------------------- #
def fit_normalize(
    input_csv: str,              # Input dataset name
    target_column: str,          # column name of target
    normalized_csv: str,         # Name of output normalized data set
    outInitalRes_json: str,      # Name of output statistics and data file
    minPercValid: float = 0.05,  # Minimum % of valid non-zero data for a column
) -> Dict[str, Any]:
    """Read, clean and z-score normalize a numeric dataset.

    Parameters
    ----------
    input_csv:
        Path of the CSV dataset. The first row must contain the column names.
    target_column:
        Name of the binary target column. It is never dropped nor normalized.
    normalized_csv:
        Path of the output CSV holding the normalized features plus the target
        column, with the original header and the original column order.
    outInitalRes_json:
        Path of the output JSON report.
    minPercValid:
        Minimum fraction (0.0-1.0) of valid values a feature must have to be
        kept. A value is valid when it is both present and different from zero.
        With ``minPercValid = 0.05`` every column with more than 95% of missing
        or zero values is dropped.

    Returns
    -------
    dict
        The same dictionary that is written to ``outInitalRes_json``.

    Raises
    ------
    FileNotFoundError
        If ``input_csv`` does not exist.
    ValueError
        If ``minPercValid`` is outside [0, 1] or ``target_column`` is missing.
    """
    if not 0.0 <= float(minPercValid) <= 1.0:
        raise ValueError(
            f"minPercValid must be a fraction between 0.0 and 1.0, got {minPercValid!r}"
        )

    input_path = Path(input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_csv}")

    # --- 1. Read the dataset ------------------------------------------------
    t0 = time.perf_counter()
    df = pd.read_csv(input_path)
    dataset_input_time = time.perf_counter() - t0

    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} not found. "
            f"Available columns: {list(df.columns)[:10]}..."
        )

    # --- 2. Process ---------------------------------------------------------
    t1 = time.perf_counter()

    original_order: List[str] = list(df.columns)
    feature_names: List[str] = [c for c in original_order if c != target_column]
    n_input_features = len(feature_names)

    # Separate the target and drop rows whose target is undefined: an unlabeled
    # sample is unusable both for training and for scoring.
    target = _normalize_target(df[target_column])
    valid_rows = target.notna()
    if not valid_rows.all():
        n_dropped_rows = int((~valid_rows).sum())
        print(
            f"[preprocessing] {n_dropped_rows} row(s) with missing target discarded.",
            file=sys.stderr,
        )
        df = df.loc[valid_rows].reset_index(drop=True)
        target = target.loc[valid_rows].reset_index(drop=True)

    features = df[feature_names]

    # Force every feature to a numeric dtype; unparsable entries become NaN and
    # are therefore treated as missing values by the rules below.
    features = features.apply(pd.to_numeric, errors="coerce")

    # 2a. Drop the empty / almost empty columns.
    dropped_feature_names = [
        name for name in feature_names if _valid_ratio(features[name]) < float(minPercValid)
    ]
    kept_feature_names = [name for name in feature_names if name not in set(dropped_feature_names)]
    features = features[kept_feature_names]

    # 2b. Z-score standardization of the kept features only.
    if kept_feature_names:
        features = features.astype(np.float64)
        means = features.mean(axis=0, skipna=True)
        stds = features.std(axis=0, ddof=0, skipna=True)

        # A constant column has std == 0: dividing by 1.0 maps it to all zeros
        # instead of producing inf/NaN. Same treatment for all-NaN columns.
        stds = stds.replace(0.0, 1.0).fillna(1.0)
        means = means.fillna(0.0)

        features = (features - means) / stds

        # Remaining missing values are imputed with the column mean, which after
        # standardization is exactly 0.0. This keeps the output free of NaNs
        # without shifting any feature distribution.
        features = features.fillna(0.0)

    # 2c. Rebuild the dataset preserving the original column order.
    normalized = features.copy()
    normalized[target_column] = target.to_numpy()
    normalized = normalized[[c for c in original_order if c in normalized.columns]]

    dataset_size = int(len(normalized))

    # --- 3. Write the outputs ----------------------------------------------
    out_data_path = _ensure_parent_dir(normalized_csv)
    normalized.to_csv(out_data_path, index=False)

    dataset_processing_time = time.perf_counter() - t1

    result: Dict[str, Any] = {
        "n_input_features": n_input_features,
        "n_kept_features": len(kept_feature_names),
        "dataset_size": dataset_size,
        "dataset_input_time": round(dataset_input_time, 4),
        "dataset_processing_time": round(dataset_processing_time, 4),
        "dropped_feature_names": dropped_feature_names,
    }

    out_json_path = _ensure_parent_dir(outInitalRes_json)
    with out_json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    return result


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preprocessing.py",
        description="Phase 1: cleaning and z-score normalization of a numeric dataset.",
    )
    parser.add_argument("--input", required=True, help="Input dataset (CSV).")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument("--out-data", required=True, help="Output normalized dataset (CSV).")
    parser.add_argument("--out-json", required=True, help="Output statistics file (JSON).")
    parser.add_argument(
        "--min-perc-valid",
        type=float,
        default=0.05,
        help="Minimum fraction (0.0-1.0) of non-zero, non-missing values "
             "required to keep a feature. Default: 0.05.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = fit_normalize(
            input_csv=args.input,
            target_column=args.target,
            normalized_csv=args.out_data,
            outInitalRes_json=args.out_json,
            minPercValid=args.min_perc_valid,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[preprocessing] Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"[preprocessing] {result['n_kept_features']}/{result['n_input_features']} "
        f"features kept, {result['dataset_size']} samples. "
        f"Read {result['dataset_input_time']}s, processing {result['dataset_processing_time']}s."
    )
    print(f"[preprocessing] Normalized dataset -> {args.out_data}")
    print(f"[preprocessing] Statistics         -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())