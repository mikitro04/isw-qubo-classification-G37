"""Streamlit dashboard for the QUBO binary classification pipeline.

Launch it from the repository root with:

    streamlit run src/qubo_project/gui.py

The GUI is a thin control layer: it collects the parameters, calls the very
same functions used by the command line interface (``fit_normalize``,
``select_features``, ``train``, ``predict``) and renders their outputs. Every
artefact is written by the backend modules into the ``outputs/`` folder at the
root of the repository.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths and imports of the project modules
# --------------------------------------------------------------------------- #
# gui.py lives in src/qubo_project/, so the repository root is two levels up.
# Everything is derived from __file__: no absolute path is hard-coded.
_HERE = Path(__file__).resolve()
ROOT_DIR = _HERE.parents[2]
SRC_DIR = _HERE.parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"

from src.qubo_project.preprocessing import fit_normalize
from src.qubo_project.feature_selection import select_features
from src.qubo_project.model import train, predict

# Output artefacts, all inside outputs/.
PATHS: Dict[str, Path] = {
    "normalized_csv": OUTPUTS_DIR / "normalized.csv",
    "preprocessing_json": OUTPUTS_DIR / "preprocessing_result.json",
    "train_csv": OUTPUTS_DIR / "training_reduced.csv",
    "test_csv": OUTPUTS_DIR / "test_reduced.csv",
    "optimizations_csv": OUTPUTS_DIR / "optimizations.csv",
    "selection_json": OUTPUTS_DIR / "feature_selection_result.json",
    "model_path": OUTPUTS_DIR / "model.joblib",
    "metrics_json": OUTPUTS_DIR / "training_metrics.json",
    "predictions_csv": OUTPUTS_DIR / "predictions.csv",
    "stats_json": OUTPUTS_DIR / "classification_stats.json",
}

PREVIEW_ROWS = 200
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # do not load huge CSVs into memory


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def rel(path: Path) -> str:
    """Path relative to the repository root, for display purposes."""
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_preview(path: Path, nrows: int = PREVIEW_ROWS) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, nrows=nrows)
    except (pd.errors.ParserError, OSError):
        return None


def read_header(path: Path) -> List[str]:
    """Column names only: cheap even on a 1.5M-row dataset."""
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except (pd.errors.ParserError, OSError, UnicodeDecodeError):
        return []


def human_size(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def download_button(path: Path, label: Optional[str] = None, key: Optional[str] = None) -> None:
    """Offer a file for download, unless it is too large to load in memory."""
    if not path.is_file():
        return
    size = path.stat().st_size
    if size > MAX_DOWNLOAD_BYTES:
        st.caption(f"{rel(path)} — {human_size(size)}: too large to download, read it from disk.")
        return
    st.download_button(
        label=label or f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/json" if path.suffix == ".json" else "text/csv",
        key=key or f"dl_{path.name}",
        use_container_width=True,
    )


def artefact_status(path: Path) -> str:
    if path.is_file():
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
        return f"✅ {rel(path)} · {human_size(path.stat().st_size)} · {stamp}"
    return f"⚪ {rel(path)} — not generated yet"


def show_error(message: str, exc: Optional[BaseException] = None) -> None:
    st.error(message)
    if exc is not None:
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def confusion_matrix_html(matrix: List[List[int]], labels: List[int]) -> str:
    """Render the confusion matrix as a coloured HTML table."""
    total = max(sum(sum(row) for row in matrix), 1)
    cells = []
    for i, row in enumerate(matrix):
        tds = [f'<th class="cm-head">real {labels[i]}</th>']
        for j, value in enumerate(row):
            share = value / total
            correct = i == j
            base = "16,150,110" if correct else "220,80,80"
            alpha = 0.12 + 0.55 * min(share * 2.2, 1.0)
            tds.append(
                f'<td style="background:rgba({base},{alpha:.2f})">'
                f'<div class="cm-value">{value:,}</div>'
                f'<div class="cm-share">{share:.1%}</div></td>'
            )
        cells.append("<tr>" + "".join(tds) + "</tr>")

    header = "".join(f'<th class="cm-head">pred {label}</th>' for label in labels)
    return f"""
    <style>
      .cm-table {{ border-collapse:separate; border-spacing:4px; width:100%;
                   font-family:inherit; text-align:center; }}
      .cm-table td {{ border-radius:8px; padding:14px 8px; }}
      .cm-head {{ font-size:0.78rem; font-weight:600; opacity:.7;
                  text-transform:uppercase; letter-spacing:.04em; padding:6px; }}
      .cm-value {{ font-size:1.35rem; font-weight:700; }}
      .cm-share {{ font-size:0.75rem; opacity:.75; }}
    </style>
    <table class="cm-table">
      <tr><th></th>{header}</tr>
      {''.join(cells)}
    </table>
    """


# --------------------------------------------------------------------------- #
# Page setup and session state
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="QUBO Binary Classification",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.55rem; }
      .phase-note { font-size: 0.88rem; opacity: 0.75; margin-bottom: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATE: Dict[str, Any] = {
    "input_csv": None,
    "target_column": None,
    "preprocessing_result": load_json(PATHS["preprocessing_json"]),
    "selection_result": load_json(PATHS["selection_json"]),
    "training_metrics": load_json(PATHS["metrics_json"]),
    "classification_stats": load_json(PATHS["stats_json"]),
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Sidebar: dataset selection and parameters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧮 QUBO Pipeline")
    st.caption("Binary classification with QUBO feature reduction")

    st.subheader("1 · Dataset")
    source = st.radio(
        "Dataset source",
        ["Choose from data/", "Upload a CSV", "Type a path"],
        label_visibility="collapsed",
    )

    input_csv: Optional[Path] = None

    if source == "Choose from data/":
        candidates = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.is_dir() else []
        if not candidates:
            st.warning(f"No CSV file found in {rel(DATA_DIR)}/.")
        else:
            chosen = st.selectbox("CSV file", [c.name for c in candidates])
            input_csv = DATA_DIR / chosen

    elif source == "Upload a CSV":
        uploaded = st.file_uploader("Select a CSV file", type=["csv"])
        if uploaded is not None:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            stored = UPLOADS_DIR / uploaded.name
            stored.write_bytes(uploaded.getbuffer())
            input_csv = stored
            st.caption(f"Saved to {rel(stored)}")

    else:
        typed = st.text_input("Path relative to the repository root", value="data/input_dataset.csv")
        if typed.strip():
            candidate = Path(typed.strip())
            input_csv = candidate if candidate.is_absolute() else ROOT_DIR / candidate

    # ---- validation of the dataset and of the target column ----------------
    columns: List[str] = []
    if input_csv is None:
        st.info("Select a dataset to enable the pipeline.")
    elif not input_csv.is_file():
        st.error(f"File not found: {rel(input_csv)}")
        input_csv = None
    else:
        columns = read_header(input_csv)
        if not columns:
            st.error("The file cannot be read as a CSV, or it has no header row.")
            input_csv = None
        else:
            st.success(f"{len(columns)} columns · {human_size(input_csv.stat().st_size)}")

    target_column: Optional[str] = None
    if columns:
        st.subheader("2 · Target column")
        # The target name is never assumed: it is picked from the real header.
        default_index = columns.index("target") if "target" in columns else len(columns) - 1
        target_column = st.selectbox("Binary target", columns, index=default_index)

    st.session_state["input_csv"] = str(input_csv) if input_csv else None
    st.session_state["target_column"] = target_column

    st.subheader("3 · Parameters")
    with st.expander("Preprocessing", expanded=True):
        min_perc_valid = st.slider(
            "Minimum fraction of valid values", 0.0, 0.50, 0.05, 0.01,
            help="Columns with fewer non-zero, non-missing values are dropped.",
        )
    with st.expander("Feature selection (QUBO)", expanded=True):
        perc_selected = st.slider("Fraction of features to select", 0.05, 1.0, 0.20, 0.01)
        allowance = st.number_input("Allowance (± features)", 0, 20, 1, 1)
        perc_test = st.slider("Test set fraction", 0.05, 0.90, 0.30, 0.05)
        alpha_computations = st.number_input("Max optimizations (alpha)", 1, 500, 100, 1)
    with st.expander("Model", expanded=True):
        classifier = st.selectbox("Classifier", ["random_forest", "logistic_regression", "decision_tree"])
        seed = st.number_input("Seed", 0, 10**6, 42, 1)

    st.divider()
    st.caption(f"Outputs → {rel(OUTPUTS_DIR)}/")


ready = st.session_state["input_csv"] is not None and st.session_state["target_column"] is not None


# --------------------------------------------------------------------------- #
# Header and progress
# --------------------------------------------------------------------------- #
st.title("QUBO Binary Classification")
st.markdown(
    '<p class="phase-note">Preprocessing → QUBO feature selection → training → '
    "classification. Each phase calls the same functions used by the CLI.</p>",
    unsafe_allow_html=True,
)

progress = st.columns(4)
phases = [
    ("Preprocessing", st.session_state["preprocessing_result"]),
    ("Feature selection", st.session_state["selection_result"]),
    ("Training", st.session_state["training_metrics"]),
    ("Prediction", st.session_state["classification_stats"]),
]
for column, (name, done) in zip(progress, phases):
    column.metric(name, "Done" if done else "Pending", delta=None)

tab_data, tab_pre, tab_qubo, tab_train, tab_pred, tab_out = st.tabs(
    ["📄 Dataset", "🧹 Preprocessing", "🧬 Feature selection", "🌲 Training",
     "🎯 Prediction", "📦 Outputs"]
)


# --------------------------------------------------------------------------- #
# Tab: dataset
# --------------------------------------------------------------------------- #
with tab_data:
    if not ready:
        st.info("Select a dataset and a target column in the sidebar.")
    else:
        input_path = Path(st.session_state["input_csv"])
        st.subheader(rel(input_path))
        preview = read_preview(input_path)
        if preview is None:
            st.error("The dataset cannot be read.")
        else:
            left, right, third = st.columns(3)
            left.metric("Columns", len(read_header(input_path)))
            right.metric("File size", human_size(input_path.stat().st_size))
            third.metric("Target column", st.session_state["target_column"])

            st.caption(f"First {len(preview)} rows")
            st.dataframe(preview, use_container_width=True, height=320)

            target_col = st.session_state["target_column"]
            values = pd.to_numeric(preview[target_col], errors="coerce").dropna()
            unique = sorted(values.unique().tolist())
            if not set(unique) <= {0, 1}:
                st.warning(
                    f"In the preview the column '{target_col}' takes the values {unique}. "
                    "The pipeline expects a binary target encoded as 0/1."
                )
            else:
                share = float((values == 1).mean()) * 100 if len(values) else 0.0
                st.caption(f"Positives in the preview: {share:.2f}%")


# --------------------------------------------------------------------------- #
# Tab: preprocessing
# --------------------------------------------------------------------------- #
with tab_pre:
    st.subheader("Phase 1 · Cleaning and z-score normalization")
    st.markdown(
        '<p class="phase-note">Drops the almost empty columns, standardizes the '
        "features and writes the normalized dataset.</p>",
        unsafe_allow_html=True,
    )

    if not ready:
        st.info("Select a dataset and a target column first.")
    elif st.button("▶ Run preprocessing", type="primary", use_container_width=True):
        input_path = Path(st.session_state["input_csv"])
        if not input_path.is_file():
            show_error(f"The input file no longer exists: {rel(input_path)}")
        else:
            try:
                with st.spinner("Reading, cleaning and normalizing…"):
                    result = fit_normalize(
                        input_csv=str(input_path),
                        target_column=st.session_state["target_column"],
                        normalized_csv=str(PATHS["normalized_csv"]),
                        outInitalRes_json=str(PATHS["preprocessing_json"]),
                        minPercValid=float(min_perc_valid),
                    )
                st.session_state["preprocessing_result"] = result
                # Downstream results are now stale.
                st.session_state["selection_result"] = None
                st.session_state["training_metrics"] = None
                st.session_state["classification_stats"] = None
                st.success("Preprocessing completed.")
            except (FileNotFoundError, ValueError) as exc:
                show_error(str(exc), exc)
            except Exception as exc:  # noqa: BLE001 - the GUI must never crash
                show_error("Unexpected error during preprocessing.", exc)

    result = st.session_state["preprocessing_result"]
    if result:
        st.divider()
        cols = st.columns(4)
        cols[0].metric("Input features", result["n_input_features"])
        cols[1].metric(
            "Kept features",
            result["n_kept_features"],
            delta=-(result["n_input_features"] - result["n_kept_features"]) or None,
        )
        cols[2].metric("Samples", f"{result['dataset_size']:,}")
        cols[3].metric("Processing time", f"{result['dataset_processing_time']:.2f}s")

        dropped = result.get("dropped_feature_names", [])
        if dropped:
            with st.expander(f"Dropped columns ({len(dropped)})"):
                st.write(", ".join(dropped))
        else:
            st.caption("No column was dropped.")

        preview = read_preview(PATHS["normalized_csv"])
        if preview is not None:
            st.caption(f"{rel(PATHS['normalized_csv'])} — first {len(preview)} rows")
            st.dataframe(preview, use_container_width=True, height=280)
        with st.expander("JSON report"):
            st.json(result)


# --------------------------------------------------------------------------- #
# Tab: feature selection
# --------------------------------------------------------------------------- #
with tab_qubo:
    st.subheader("Phase 2 · QUBO feature selection")
    st.markdown(
        '<p class="phase-note">Builds Q from the Spearman correlations of the '
        "training set and varies α until K features are selected.</p>",
        unsafe_allow_html=True,
    )

    if not PATHS["normalized_csv"].is_file():
        st.info("Run the preprocessing first: the normalized dataset is missing.")
    elif st.button("▶ Run feature selection", type="primary", use_container_width=True):
        try:
            with st.spinner("Building Q and optimizing over α…"):
                result = select_features(
                    normalized_csv=str(PATHS["normalized_csv"]),
                    reducedTrain_csv=str(PATHS["train_csv"]),
                    reducedTest_csv=str(PATHS["test_csv"]),
                    output_ottim_csv=str(PATHS["optimizations_csv"]),
                    output_json=str(PATHS["selection_json"]),
                    target_column=st.session_state["target_column"],
                    percTest=float(perc_test),
                    allowance=int(allowance),
                    seed=int(seed),
                    percSelected=float(perc_selected),
                    alpha_computations=int(alpha_computations),
                )
            st.session_state["selection_result"] = result
            st.session_state["training_metrics"] = None
            st.session_state["classification_stats"] = None

            hit = abs(result["n_selected"] - result["target_k"]) <= result["allowance"]
            if hit:
                st.success("Feature selection completed.")
            else:
                st.warning(
                    f"{result['n_selected']} features selected, target was "
                    f"{result['target_k']} ± {result['allowance']}. "
                    "Try increasing the maximum number of optimizations."
                )
        except (FileNotFoundError, ValueError) as exc:
            show_error(str(exc), exc)
        except Exception as exc:  # noqa: BLE001
            show_error("Unexpected error during feature selection.", exc)

    result = st.session_state["selection_result"]
    if result:
        st.divider()
        cols = st.columns(5)
        cols[0].metric(
            "Selected features",
            result["n_selected"],
            delta=result["n_selected"] - result["target_k"] or None,
            help=f"Target: {result['target_k']} ± {result['allowance']}",
        )
        cols[1].metric("Available features", result["n_features"])
        cols[2].metric("α found", f"{result['alpha']:.4f}")
        cols[3].metric("Optimizations", result["alpha_computations"])
        cols[4].metric("Q build time", f"{result['q_matrix_creation_time']:.2f}s")

        cols = st.columns(4)
        cols[0].metric("Training samples", f"{result['training_dataset_size']:,}")
        cols[1].metric("Test samples", f"{result['test_dataset_size']:,}")
        cols[2].metric("Mean opt. time", f"{result['mean_optimization_time']:.3f}s")
        cols[3].metric("Std dev opt. time", f"{result['std_dev_optimization_time']:.3f}s")

        history = read_preview(PATHS["optimizations_csv"], nrows=1000)
        if history is not None and not history.empty:
            left, right = st.columns(2)
            with left:
                st.caption("Selected features vs α")
                st.line_chart(history.set_index("alpha")["n_features_selected"], height=260)
            with right:
                st.caption("Cost function vs α")
                st.line_chart(history.set_index("alpha")["cost_function_value"], height=260)
            with st.expander("Optimization history"):
                st.dataframe(history, use_container_width=True, hide_index=True)

        with st.expander(f"Selected features ({result['n_selected']})", expanded=True):
            st.write(", ".join(result["selected_feature_names"]))

        left, right = st.columns(2)
        for column, path, label in (
            (left, PATHS["train_csv"], "Reduced training set"),
            (right, PATHS["test_csv"], "Reduced test set"),
        ):
            preview = read_preview(path, nrows=50)
            if preview is not None:
                column.caption(f"{label} — {rel(path)}")
                column.dataframe(preview, use_container_width=True, height=240)

        with st.expander("JSON report"):
            st.json(result)


# --------------------------------------------------------------------------- #
# Tab: training
# --------------------------------------------------------------------------- #
with tab_train:
    st.subheader("Phase 3 · Classifier training")
    st.markdown(
        '<p class="phase-note">Trains the selected classifier on the reduced '
        "training set and saves the model with joblib.</p>",
        unsafe_allow_html=True,
    )

    if not PATHS["train_csv"].is_file():
        st.info("Run the feature selection first: the reduced training set is missing.")
    else:
        st.caption(f"Classifier: **{classifier}** · seed: **{seed}**")
        if st.button("▶ Train the model", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Training {classifier}…"):
                    metrics = train(
                        classifier=classifier,
                        reducedTrain_csv=str(PATHS["train_csv"]),
                        target_column=st.session_state["target_column"],
                        model_path=str(PATHS["model_path"]),
                        metrics_json=str(PATHS["metrics_json"]),
                        seed=int(seed),
                    )
                st.session_state["training_metrics"] = metrics
                st.session_state["classification_stats"] = None
                st.success(f"Model saved to {rel(PATHS['model_path'])}")
            except (FileNotFoundError, ValueError) as exc:
                show_error(str(exc), exc)
            except Exception as exc:  # noqa: BLE001
                show_error("Unexpected error during training.", exc)

    metrics = st.session_state["training_metrics"]
    if metrics:
        st.divider()
        cols = st.columns(5)
        cols[0].metric("Classifier", metrics["classifier"])
        cols[1].metric("Samples", f"{metrics['n_samples']:,}")
        cols[2].metric("Features", metrics["n_features"])
        cols[3].metric("Positives", f"{metrics['target_1_percentage']:.2f}%")
        cols[4].metric("Training time", f"{metrics['training_time']:.2f}s")

        if float(metrics["target_1_percentage"]) < 5.0:
            st.info(
                "Strongly imbalanced target: the F1 of class 1 is the metric to watch, "
                "accuracy alone is misleading."
            )

        if PATHS["model_path"].is_file():
            st.caption(artefact_status(PATHS["model_path"]))
        with st.expander("JSON report"):
            st.json(metrics)


# --------------------------------------------------------------------------- #
# Tab: prediction
# --------------------------------------------------------------------------- #
with tab_pred:
    st.subheader("Phase 4 · Classification of the test set")
    st.markdown(
        '<p class="phase-note">Applies the saved model to the reduced test set '
        "and computes the quality statistics.</p>",
        unsafe_allow_html=True,
    )

    missing = [
        rel(path) for path in (PATHS["model_path"], PATHS["test_csv"]) if not path.is_file()
    ]
    if missing:
        st.info("Missing artefacts: " + ", ".join(missing))
    else:
        target_options = [PATHS["test_csv"], PATHS["train_csv"]]
        chosen = st.selectbox(
            "Dataset to classify",
            [rel(p) for p in target_options if p.is_file()],
            help="Any reduced dataset can be classified, not only the test set.",
        )
        if st.button("▶ Run prediction", type="primary", use_container_width=True):
            try:
                with st.spinner("Classifying…"):
                    stats = predict(
                        reduced_Test_csv=str(ROOT_DIR / chosen),
                        target_column=st.session_state["target_column"],
                        model_path=str(PATHS["model_path"]),
                        predictions_csv=str(PATHS["predictions_csv"]),
                        classif_stats_json=str(PATHS["stats_json"]),
                    )
                st.session_state["classification_stats"] = stats
                st.success("Classification completed.")
            except (FileNotFoundError, ValueError) as exc:
                show_error(str(exc), exc)
            except Exception as exc:  # noqa: BLE001
                show_error("Unexpected error during prediction.", exc)

    stats = st.session_state["classification_stats"]
    if stats:
        st.divider()
        cols = st.columns(5)
        cols[0].metric("Samples", f"{stats['n_samples']:,}")
        cols[1].metric("Accuracy", f"{stats['accuracy']:.4f}")
        cols[2].metric("F1 (class 1)", f"{stats['class_1']['f1']:.4f}")
        roc = stats.get("roc_auc")
        cols[3].metric("ROC-AUC", f"{roc:.4f}" if roc is not None else "n/a")
        cols[4].metric("Positives", f"{stats['target_1_percentage']:.2f}%")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Confusion matrix**")
            matrix = stats["confusion_matrix"]
            st.markdown(
                confusion_matrix_html(matrix["matrix"], matrix["labels"]),
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("**Per-class metrics**")
            per_class = pd.DataFrame(
                [
                    {
                        "class": label,
                        "precision": stats[f"class_{label}"]["precision"],
                        "recall": stats[f"class_{label}"]["recall"],
                        "f1": stats[f"class_{label}"]["f1"],
                        "support": stats[f"class_{label}"]["support"],
                    }
                    for label in (0, 1)
                ]
            )
            st.dataframe(
                per_class.style.format(
                    {"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        predictions = read_preview(PATHS["predictions_csv"], nrows=5000)
        if predictions is not None and not predictions.empty:
            st.markdown("**Score distribution (positive class)**")
            counts, edges = np.histogram(
                predictions["score"].to_numpy(), bins=20, range=(0.0, 1.0)
            )
            st.bar_chart(
                pd.DataFrame(
                    {"count": counts},
                    index=pd.Index(np.round(edges[:-1], 2), name="score"),
                ),
                height=220,
            )
            st.caption(f"{rel(PATHS['predictions_csv'])} — first {len(predictions)} rows")
            st.dataframe(
                predictions.head(PREVIEW_ROWS), use_container_width=True,
                height=280, hide_index=True,
            )

        with st.expander("JSON report"):
            st.json(stats)


# --------------------------------------------------------------------------- #
# Tab: outputs
# --------------------------------------------------------------------------- #
with tab_out:
    st.subheader("Generated artefacts")
    st.caption(f"All files are written by the backend modules into {rel(OUTPUTS_DIR)}/")

    order = [
        ("Normalized dataset", PATHS["normalized_csv"]),
        ("Preprocessing report", PATHS["preprocessing_json"]),
        ("Reduced training set", PATHS["train_csv"]),
        ("Reduced test set", PATHS["test_csv"]),
        ("Optimization history", PATHS["optimizations_csv"]),
        ("Feature selection report", PATHS["selection_json"]),
        ("Trained model", PATHS["model_path"]),
        ("Training metrics", PATHS["metrics_json"]),
        ("Predictions", PATHS["predictions_csv"]),
        ("Classification statistics", PATHS["stats_json"]),
    ]

    for label, path in order:
        col_info, col_button = st.columns([3, 1])
        col_info.markdown(f"**{label}**")
        col_info.caption(artefact_status(path))
        with col_button:
            download_button(path, label="Download", key=f"out_{path.name}")

    st.divider()
    if st.button("🗑 Clear the pipeline state (files are kept)"):
        for key in (
            "preprocessing_result", "selection_result",
            "training_metrics", "classification_stats",
        ):
            st.session_state[key] = None
        st.rerun()


# --------------------------------------------------------------------------- #
# Guard: this script is meant to be launched through Streamlit
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    try:
        from streamlit.runtime import exists as _runtime_exists

        if not _runtime_exists():
            print(
                "This module is a Streamlit application. Launch it with:\n\n"
                "    streamlit run src/qubo_project/gui.py\n"
            )
    except ImportError:
        pass
