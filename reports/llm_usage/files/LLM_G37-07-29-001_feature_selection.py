"""QUBO-based feature selection for the binary classification project.

Phase 2 of the pipeline:

1. read the normalized dataset produced by ``preprocessing.py``;
2. split it with a clean cut into training set (first M samples) and test set;
3. compute the absolute Spearman correlations **on the training set only**;
4. build the QUBO matrix Q(alpha) and minimize f(x) = -x^T Q x;
5. vary alpha until the optimum selects K = round(percSelected * n) +/- allowance
   features, within a maximum number of optimizations;
6. write the reduced training/test datasets, the alpha history and a JSON report.

The module is importable and executable from the command line:

    python feature_selection.py \\
        --in-normalized normalized.csv \\
        --out-train training_reduced.csv \\
        --out-test test_reduced.csv \\
        --out-optimizations optimizations.csv \\
        --out-json feature_selection_result.json \\
        --target target \\
        --perc-selected 0.20 \\
        --allowance 1 \\
        --perc-test 0.30 \\
        --seed 42 \\
        --alpha-computations 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# The QUBO solver is optional: if dimod/neal are not installed the module falls
# back to an internal simulated annealing implementation, so the project always
# runs from a bare `pip install -r requirements.txt`.
try:  # pragma: no cover - depends on the environment
    import dimod
    import neal

    _HAS_NEAL = True
except ImportError:  # pragma: no cover
    _HAS_NEAL = False

__all__ = ["select_features", "main"]

# Sampling effort of the annealer. Kept module level so the values are easy to
# tune and to document in the final report.
_NUM_READS = 20
_NUM_SWEEPS = 400
_FALLBACK_NUM_READS = 8
_FALLBACK_NUM_SWEEPS = 250


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_parent_dir(path: str) -> Path:
    """Create the parent directory of ``path`` if needed (no absolute paths)."""
    file_path = Path(path)
    if file_path.parent and str(file_path.parent) not in ("", "."):
        file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _spearman_blocks(
    features: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(|rho_Vj|, |rho_jk|)`` computed with the Spearman formula.

    Spearman's rho is Pearson's r applied to the ranks, so the ranks are
    computed once per column and a single correlation matrix is derived from
    them. This is much cheaper than a pairwise Spearman on every column pair,
    which matters on the 1.5M-sample evaluation dataset.

    The diagonal of the returned feature-feature matrix is zeroed, since the
    QUBO cost function sums over ``k != j`` only.
    """
    n_samples, n_features = features.shape

    # Average ranks (ties are handled the same way scipy.stats.rankdata does).
    ranked = np.empty((n_samples, n_features + 1), dtype=np.float64)
    stacked = np.column_stack([features, target])
    for col in range(n_features + 1):
        ranked[:, col] = pd.Series(stacked[:, col]).rank(method="average").to_numpy()
    del stacked

    # Pearson on the ranks. Constant columns give a 0/0 -> NaN, replaced by 0.0:
    # a feature with no variance carries no information and will not be picked.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(ranked, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    del ranked

    rho_target = np.abs(corr[:n_features, n_features])
    rho_features = np.abs(corr[:n_features, :n_features])
    np.fill_diagonal(rho_features, 0.0)
    return rho_target, rho_features


def _build_q(alpha: float, rho_target: np.ndarray, rho_features: np.ndarray) -> np.ndarray:
    """Assemble the QUBO matrix Q for a given alpha.

    Diagonal:      Q_jj = alpha * |rho_Vj|
    Off-diagonal:  Q_jk = -(1 - alpha) * |rho_jk|   (both triangles, so that
                   x^T Q x reproduces the double sum over j != k)
    """
    q = -(1.0 - alpha) * rho_features
    np.fill_diagonal(q, alpha * rho_target)
    return q


def _cost(x: np.ndarray, q: np.ndarray) -> float:
    """Objective value f(x) = -x^T Q x."""
    return float(-(x @ q @ x))


def _solve_with_neal(q: np.ndarray, seed: int) -> np.ndarray:
    """Minimize f(x) = -x^T Q x with dimod/neal simulated annealing.

    The BQM is expressed in upper-triangular form:
        linear_j    = -Q_jj
        quadratic_jk = -2 * Q_jk   for j < k
    so that its energy equals -x^T Q x exactly.
    """
    upper = -2.0 * np.triu(q, k=1)
    np.fill_diagonal(upper, -np.diag(q))
    bqm = dimod.BinaryQuadraticModel(upper, dimod.BINARY)

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        bqm, num_reads=_NUM_READS, num_sweeps=_NUM_SWEEPS, seed=seed
    )
    best = sampleset.first.sample
    return np.array([best[i] for i in range(q.shape[0])], dtype=np.int8)


def _solve_with_numpy(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fallback simulated annealing, used when dimod/neal are unavailable.

    Single-bit flips with incremental energy updates: flipping bit i changes
    the energy by ``-d * (Q_ii + 2 * g_i)`` where ``d`` is +1 when the bit is
    switched on and ``g_i`` is the interaction of i with the other active bits.
    Keeping ``h = Q @ x`` up to date makes each flip O(n) instead of O(n^2).
    """
    n = q.shape[0]
    diag = np.diag(q).copy()
    scale = float(np.abs(q).max()) or 1.0
    beta_schedule = np.geomspace(0.1 / scale, 50.0 / scale, _FALLBACK_NUM_SWEEPS)

    best_x, best_energy = None, np.inf
    for _ in range(_FALLBACK_NUM_READS):
        x = rng.integers(0, 2, size=n).astype(np.int8)
        h = q @ x
        energy = -float(x @ h)

        for beta in beta_schedule:
            for i in rng.permutation(n):
                d = 1 - 2 * x[i]                       # +1 turn on, -1 turn off
                g = h[i] - diag[i] * x[i]
                delta = -d * (diag[i] + 2.0 * g)
                if delta <= 0.0 or rng.random() < np.exp(-beta * delta):
                    x[i] += d
                    h += d * q[:, i]
                    energy += delta

        if energy < best_energy:
            best_energy, best_x = energy, x.copy()

    return best_x


def _optimize(q: np.ndarray, seed: int, rng: np.random.Generator) -> np.ndarray:
    if _HAS_NEAL:
        return _solve_with_neal(q, seed)
    return _solve_with_numpy(q, rng)


# --------------------------------------------------------------------------- #
# Mandatory interface
# --------------------------------------------------------------------------- #
def select_features(
    normalized_csv: str,           # Input dataset name
    reducedTrain_csv: str,         # Name of output training dataset with reduced feat.
    reducedTest_csv: str,          # Name of output test dataset with reduced features
    output_ottim_csv: str,         # Name of output optimization data varying alpha
    output_json: str,              # Name of output statistics and data file
    target_column: str,            # Column name of target
    percTest: float = 0.30,        # % of test data with respect to the dataset size
    allowance: int = 1,            # Allowance of features to select
    seed: int = 42,                # Seed for random repeatibility
    percSelected: float = 0.20,    # percentage of features to select
    alpha_computations: int = 100  # Max. n. of optimizations varying alpha
) -> Dict[str, Any]:
    """Select a subset of features by solving a QUBO problem.

    The number of selected features is not constrained inside the QUBO: alpha
    is varied (bisection) until the optimum happens to contain K ones, with
    K = round(percSelected * n_features) +/- allowance.

    Returns
    -------
    dict
        The same dictionary written to ``output_json``.
    """
    if not 0.0 < float(percTest) < 1.0:
        raise ValueError(f"percTest must be strictly between 0 and 1, got {percTest!r}")
    if not 0.0 < float(percSelected) <= 1.0:
        raise ValueError(
            f"percSelected must be in (0, 1], got {percSelected!r}"
        )
    if int(allowance) < 0:
        raise ValueError(f"allowance must be non-negative, got {allowance!r}")
    if int(alpha_computations) < 1:
        raise ValueError(
            f"alpha_computations must be at least 1, got {alpha_computations!r}"
        )

    input_path = Path(normalized_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Normalized dataset not found: {normalized_csv}")

    rng = np.random.default_rng(seed)

    # --- 1. Read and split with a clean cut ---------------------------------
    df = pd.read_csv(input_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found in {normalized_csv}")

    feature_names: List[str] = [c for c in df.columns if c != target_column]
    n_features = len(feature_names)
    if n_features == 0:
        raise ValueError("The normalized dataset contains no feature column.")

    n_total = len(df)
    n_test = int(round(float(percTest) * n_total))
    n_test = min(max(n_test, 1), n_total - 1)  # both splits must be non-empty
    n_train = n_total - n_test                 # M: first M samples are training

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    # --- 2. Spearman correlations, TRAINING SET ONLY ------------------------
    t0 = time.perf_counter()
    train_features = train_df[feature_names].to_numpy(dtype=np.float64)
    train_target = train_df[target_column].to_numpy(dtype=np.float64)

    if np.unique(train_target).size < 2:
        print(
            "[feature_selection] Warning: the training target is constant, "
            "all feature-target correlations are 0.",
            file=sys.stderr,
        )

    rho_target, rho_features = _spearman_blocks(train_features, train_target)
    del train_features
    q_matrix_creation_time = time.perf_counter() - t0

    # --- 3. Search over alpha ----------------------------------------------
    target_k = int(round(float(percSelected) * n_features))
    target_k = min(max(target_k, 1), n_features)
    k_min = max(target_k - int(allowance), 0)
    k_max = min(target_k + int(allowance), n_features)

    history: List[Dict[str, float]] = []
    optimization_times: List[float] = []

    lo, hi = 0.0, 1.0          # n_selected(alpha) is monotonically increasing
    alpha = 0.5
    best_x: np.ndarray | None = None
    best_alpha = alpha
    best_cost = float("nan")
    best_distance = np.inf

    for _ in range(int(alpha_computations)):
        q = _build_q(alpha, rho_target, rho_features)

        t_opt = time.perf_counter()
        x = _optimize(q, seed, rng)
        elapsed = time.perf_counter() - t_opt

        n_selected = int(x.sum())
        cost = _cost(x.astype(np.float64), q)

        optimization_times.append(elapsed)
        history.append(
            {
                "alpha": alpha,
                "optimization_time": elapsed,
                "n_features_selected": n_selected,
                "cost_function_value": cost,
            }
        )

        # Keep the closest attempt so far; ties are broken by the lower cost.
        distance = abs(n_selected - target_k)
        if distance < best_distance or (distance == best_distance and cost < best_cost):
            best_distance, best_x, best_alpha, best_cost = distance, x.copy(), alpha, cost

        if k_min <= n_selected <= k_max:
            break

        if n_selected < target_k:
            lo = alpha       # too few features -> more weight on influence
        else:
            hi = alpha       # too many features -> more weight on independence

        next_alpha = 0.5 * (lo + hi)
        if abs(next_alpha - alpha) < 1e-12:
            break            # the interval collapsed, no further refinement
        alpha = next_alpha

    if best_x is None:  # pragma: no cover - alpha_computations >= 1 guarantees a run
        raise RuntimeError("No optimization was performed.")

    selected_vector = best_x.astype(int).tolist()
    selected_feature_names = [
        name for name, bit in zip(feature_names, selected_vector) if bit == 1
    ]
    n_selected = len(selected_feature_names)

    if not (k_min <= n_selected <= k_max):
        print(
            f"[feature_selection] Warning: {n_selected} features selected, "
            f"target was {target_k} +/- {allowance} after {len(history)} "
            f"optimizations. Closest result kept (alpha = {best_alpha:.6f}).",
            file=sys.stderr,
        )

    # --- 4. Reduce and save the two datasets --------------------------------
    # Selected features keep their original order; the target is the last column.
    kept_columns = selected_feature_names + [target_column]
    train_out = _ensure_parent_dir(reducedTrain_csv)
    test_out = _ensure_parent_dir(reducedTest_csv)
    train_df[kept_columns].to_csv(train_out, index=False)
    test_df[kept_columns].to_csv(test_out, index=False)

    # --- 5. Save the alpha history (ascending alpha) ------------------------
    hist_df = pd.DataFrame(
        history,
        columns=["alpha", "optimization_time", "n_features_selected", "cost_function_value"],
    ).sort_values("alpha", kind="stable")
    hist_out = _ensure_parent_dir(output_ottim_csv)
    hist_df.to_csv(hist_out, index=False)

    # --- 6. Save the JSON report -------------------------------------------
    result: Dict[str, Any] = {
        "n_features": n_features,
        "target_ratio": float(percSelected),
        "target_k": target_k,
        "allowance": int(allowance),
        "n_selected": n_selected,
        "alpha": round(float(best_alpha), 6),
        "selected_vector": selected_vector,
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing" if _HAS_NEAL else "simulated_annealing_numpy",
        "seed": int(seed),
        "alpha_computations": len(history),
        "percTest": float(percTest),
        "training_dataset_size": int(n_train),
        "test_dataset_size": int(n_test),
        "q_matrix_creation_time": round(q_matrix_creation_time, 4),
        "mean_optimization_time": round(float(np.mean(optimization_times)), 4),
        "std_dev_optimization_time": round(float(np.std(optimization_times)), 4),
    }

    json_out = _ensure_parent_dir(output_json)
    with json_out.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    return result


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature_selection.py",
        description="Phase 2: QUBO feature selection and train/test split.",
    )
    parser.add_argument("--in-normalized", required=True, help="Normalized dataset (CSV).")
    parser.add_argument("--out-train", required=True, help="Reduced training set (CSV).")
    parser.add_argument("--out-test", required=True, help="Reduced test set (CSV).")
    parser.add_argument(
        "--out-optimizations", required=True, help="History of the alpha optimizations (CSV)."
    )
    parser.add_argument("--out-json", required=True, help="Output statistics file (JSON).")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument(
        "--perc-selected", type=float, default=0.20,
        help="Fraction of features to select. Default: 0.20.",
    )
    parser.add_argument(
        "--allowance", type=int, default=1,
        help="Tolerance, in number of features, around the target K. Default: 1.",
    )
    parser.add_argument(
        "--perc-test", type=float, default=0.30,
        help="Fraction of samples used as test set. Default: 0.30.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument(
        "--alpha-computations", type=int, default=100,
        help="Maximum number of optimizations while varying alpha. Default: 100.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = select_features(
            normalized_csv=args.in_normalized,
            reducedTrain_csv=args.out_train,
            reducedTest_csv=args.out_test,
            output_ottim_csv=args.out_optimizations,
            output_json=args.out_json,
            target_column=args.target,
            percTest=args.perc_test,
            allowance=args.allowance,
            seed=args.seed,
            percSelected=args.perc_selected,
            alpha_computations=args.alpha_computations,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[feature_selection] Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"[feature_selection] {result['n_selected']}/{result['n_features']} features "
        f"selected (target {result['target_k']} +/- {result['allowance']}) "
        f"at alpha = {result['alpha']} after {result['alpha_computations']} optimizations."
    )
    print(f"[feature_selection] Reduced training set -> {args.out_train}")
    print(f"[feature_selection] Reduced test set     -> {args.out_test}")
    print(f"[feature_selection] Alpha history        -> {args.out_optimizations}")
    print(f"[feature_selection] Statistics           -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())