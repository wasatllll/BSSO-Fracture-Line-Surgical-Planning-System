# -*- coding: utf-8 -*-
"""
Streamlit app: BSSO lingual fracture-line reverse-planning system.

This deployment does NOT require the original training dataset.
Class-specific empirical domains are pre-computed and hard-coded below.

Required files in the app directory:
    app.py
    result/best_model.pkl
    result/scaler.pkl
    result/label_encoder.pkl

Run:
    streamlit run app.py
"""

import os
from collections import deque

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# 1. Page configuration
# ============================================================

st.set_page_config(
    page_title="BSSO Lingual Fracture-Line Planning System",
    page_icon="🦷",
    layout="wide"
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# 2. Paths and fixed configuration
# ============================================================

MODEL_PATH = "result/best_model.pkl"
SCALER_PATH = "result/scaler.pkl"
LABEL_ENCODER_PATH = "result/label_encoder.pkl"

DEFAULT_FEATURES = ["LLBCE", "PMBT", "MRT", "Depth  of A"]
ROBUST_LEVELS = [1, 2]
NEAR_OPTIMAL_RATIO = 0.90

# ------------------------------------------------------------
# IMPORTANT:
# These ranges replace the original training dataset in deployment.
# They should be the pre-computed class-specific empirical ranges
# from the training cohort.
#
# Level 1 = subtype-specific IQR domain, strict recommendation
# Level 2 = subtype-specific 10th-90th percentile domain, strict recommendation
# ------------------------------------------------------------
CLASS_DOMAINS = {
    "1": {
        "level1": {
            "name": "Level 1: IQR strict domain",
            "depth_range": (1.328, 1.900),
            "llbce_range": (0.625, 1.658),
        },
        "level2": {
            "name": "Level 2: 10th-90th percentile strict domain",
            "depth_range": (1.177, 2.275),
            "llbce_range": (-0.963, 2.134),
        },
    },
    "2": {
        "level1": {
            "name": "Level 1: IQR strict domain",
            "depth_range": (2.085, 2.460),
            "llbce_range": (-1.015, 1.300),
        },
        "level2": {
            "name": "Level 2: 10th-90th percentile strict domain",
            "depth_range": (1.880, 2.830),
            "llbce_range": (-1.610, 1.530),
        },
    },
    "3": {
        "level1": {
            "name": "Level 1: IQR strict domain",
            "depth_range": (1.330, 1.720),
            "llbce_range": (1.982, 3.918),
        },
        "level2": {
            "name": "Level 2: 10th-90th percentile strict domain",
            "depth_range": (1.150, 1.870),
            "llbce_range": (1.094, 4.884),
        },
    },
}


# ============================================================
# 3. Utility functions
# ============================================================

def normalize_name(name: str) -> str:
    return (
        str(name)
        .replace("\u00a0", " ")
        .replace("\t", " ")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace(".", "")
    )


def clean_type_label(x) -> str:
    """Convert 1, 1.0, 'Type 1', 'TYPE1' to '1'."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace("TYPE", "").replace("Type", "").replace("type", "")
    s = s.replace(" ", "")
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s


def is_depth_feature(name: str) -> bool:
    return normalize_name(name) in {
        normalize_name("Depth  of A"),
        normalize_name("Depth of A"),
        normalize_name("Depth.of.A"),
        normalize_name("Depth_of_A"),
    }


def is_llbce_feature(name: str) -> bool:
    return normalize_name(name) == normalize_name("LLBCE")


def is_pmbt_feature(name: str) -> bool:
    return normalize_name(name) == normalize_name("PMBT")


def is_mrt_feature(name: str) -> bool:
    return normalize_name(name) == normalize_name("MRT")


def connected_component_from_best(mask: np.ndarray, best_pos: tuple[int, int]) -> np.ndarray:
    """Return the 4-neighbour connected component containing best_pos."""
    if not mask[best_pos]:
        return np.zeros_like(mask, dtype=bool)

    visited = np.zeros_like(mask, dtype=bool)
    q = deque([best_pos])
    visited[best_pos] = True

    n_row, n_col = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_row and 0 <= cc < n_col:
                if mask[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True
                    q.append((rr, cc))
    return visited


def align_proba_to_classes(model, proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Align predict_proba columns to encoded class order 0, 1, 2 when possible."""
    proba = np.asarray(proba)
    if proba.ndim == 3:
        proba = np.squeeze(proba)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba output has invalid shape: {proba.shape}")
    if proba.shape[1] != n_classes:
        raise ValueError(
            f"Probability columns ({proba.shape[1]}) do not match number of classes ({n_classes})."
        )
    if not hasattr(model, "classes_"):
        return proba

    try:
        model_classes_int = np.asarray(model.classes_).astype(int)
        if list(model_classes_int) == list(range(n_classes)):
            return proba

        aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
        for j, cls in enumerate(model_classes_int):
            if 0 <= cls < n_classes:
                aligned[:, cls] = proba[:, j]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned / row_sums
    except Exception:
        return proba


def safe_fmt(x, digits=2):
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


# ============================================================
# 4. Model loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, scaler_path: str, encoder_path: str):
    missing = [p for p in [model_path, scaler_path, encoder_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    class_names = [clean_type_label(x) for x in label_encoder.classes_]

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = DEFAULT_FEATURES

    return model, scaler, label_encoder, class_names, feature_names


# ============================================================
# 5. Reverse-planning engine
# ============================================================

class BSSOReversePlanner:
    def __init__(self, model, scaler, class_names, feature_names, grid_size=120):
        self.model = model
        self.scaler = scaler
        self.class_names = class_names
        self.feature_names = feature_names
        self.grid_size = int(grid_size)

    def build_grid(self, depth_range, llbce_range, mrt_value, pmbt_value):
        depth_values = np.linspace(depth_range[0], depth_range[1], self.grid_size)
        llbce_values = np.linspace(llbce_range[0], llbce_range[1], self.grid_size)
        depth_grid, llbce_grid = np.meshgrid(depth_values, llbce_values)

        grid_df = pd.DataFrame(index=np.arange(depth_grid.size))
        for f in self.feature_names:
            if is_llbce_feature(f):
                grid_df[f] = llbce_grid.ravel()
            elif is_pmbt_feature(f):
                grid_df[f] = float(pmbt_value)
            elif is_mrt_feature(f):
                grid_df[f] = float(mrt_value)
            elif is_depth_feature(f):
                grid_df[f] = depth_grid.ravel()
            else:
                raise ValueError(
                    f"Unrecognized feature name in scaler/model: {f}. "
                    "Expected LLBCE, PMBT, MRT, and Depth of A."
                )
        grid_df = grid_df[self.feature_names]
        return grid_df, depth_grid, llbce_grid

    def predict_grid(self, grid_df):
        grid_scaled = self.scaler.transform(grid_df)
        proba = self.model.predict_proba(grid_scaled)
        proba = align_proba_to_classes(self.model, proba, len(self.class_names))
        top_class_grid = np.argmax(proba, axis=1).reshape(self.grid_size, self.grid_size)
        prob_grids = [proba[:, k].reshape(self.grid_size, self.grid_size) for k in range(len(self.class_names))]
        return proba, prob_grids, top_class_grid

    def evaluate_domain(self, class_label, level_id, domain, mrt_value, pmbt_value, strict=True):
        class_label = clean_type_label(class_label)
        class_idx = self.class_names.index(class_label)

        grid_df, depth_grid, llbce_grid = self.build_grid(
            depth_range=domain["depth_range"],
            llbce_range=domain["llbce_range"],
            mrt_value=mrt_value,
            pmbt_value=pmbt_value,
        )
        proba, prob_grids, top_class_grid = self.predict_grid(grid_df)
        p_target = prob_grids[class_idx]

        if strict:
            valid_for_best = top_class_grid == class_idx
            if not np.any(valid_for_best):
                best_pos = np.unravel_index(np.argmax(p_target), p_target.shape)
                return {
                    "success": False,
                    "reason": "No grid point predicted the target type as the top class.",
                    "class_label": class_label,
                    "class_idx": class_idx,
                    "level": level_id,
                    "domain_name": domain["name"],
                    "depth_grid": depth_grid,
                    "llbce_grid": llbce_grid,
                    "prob_grids": prob_grids,
                    "p_target": p_target,
                    "top_class_grid": top_class_grid,
                    "best_pos": best_pos,
                    "stable_mask": np.zeros_like(p_target, dtype=bool),
                    "max_probability": float(p_target[best_pos]),
                    "near_cutoff": np.nan,
                    "proba_at_optimal": proba[np.ravel_multi_index(best_pos, p_target.shape), :],
                }
            p_search = p_target.copy()
            p_search[~valid_for_best] = -np.inf
            best_pos = np.unravel_index(np.argmax(p_search), p_search.shape)
        else:
            best_pos = np.unravel_index(np.argmax(p_target), p_target.shape)

        pmax = float(p_target[best_pos])
        near_cutoff = NEAR_OPTIMAL_RATIO * pmax
        near_mask = p_target >= near_cutoff
        top_mask = (top_class_grid == class_idx) if strict else np.ones_like(p_target, dtype=bool)
        candidate_mask = near_mask & top_mask
        stable_mask = connected_component_from_best(candidate_mask, best_pos)
        success = bool(np.any(stable_mask))

        return {
            "success": success,
            "reason": "Stable connected region found." if success else "No stable connected region.",
            "class_label": class_label,
            "class_idx": class_idx,
            "level": level_id,
            "domain_name": domain["name"],
            "depth_grid": depth_grid,
            "llbce_grid": llbce_grid,
            "prob_grids": prob_grids,
            "p_target": p_target,
            "top_class_grid": top_class_grid,
            "best_pos": best_pos,
            "stable_mask": stable_mask,
            "max_probability": pmax,
            "near_cutoff": near_cutoff,
            "proba_at_optimal": proba[np.ravel_multi_index(best_pos, p_target.shape), :],
        }

    def robust_search_for_class(self, class_label, mrt_value, pmbt_value):
        """Try Level 1 first; if failed, try Level 2. No Level 3/4 output is used as recommendation."""
        domains = CLASS_DOMAINS[class_label]
        attempts = []

        result_l1 = self.evaluate_domain(class_label, level_id=1, domain=domains["level1"], mrt_value=mrt_value, pmbt_value=pmbt_value, strict=True)
        attempts.append(result_l1)
        if result_l1["success"]:
            return result_l1, attempts

        result_l2 = self.evaluate_domain(class_label, level_id=2, domain=domains["level2"], mrt_value=mrt_value, pmbt_value=pmbt_value, strict=True)
        attempts.append(result_l2)
        if result_l2["success"]:
            return result_l2, attempts

        # Return Level 2 surface for visualization, but mark as not robust.
        return result_l2, attempts

    def summarize_result(self, result):
        depth_grid = result["depth_grid"]
        llbce_grid = result["llbce_grid"]
        p_target = result["p_target"]
        stable_mask = result["stable_mask"]
        best_pos = result["best_pos"]

        class_label = result["class_label"]
        summary = {
            "Type": f"Type {class_label}",
            "Selected level": result["level"] if result["success"] else "No robust Level 1/2",
            "Domain": result["domain_name"],
            "Max P(target)": result["max_probability"],
            "Optimal Depth of A": float(depth_grid[best_pos]),
            "Optimal LLBCE": float(llbce_grid[best_pos]),
            "Stable points": int(np.sum(stable_mask)),
            "Stable area, %": float(np.sum(stable_mask) / p_target.size * 100),
            "Robust recommendation": bool(result["success"]),
        }

        for k, cname in enumerate(self.class_names):
            summary[f"P(Type {cname}) at optimal"] = float(result["proba_at_optimal"][k])

        if result["success"]:
            stable_depth = depth_grid[stable_mask]
            stable_llbce = llbce_grid[stable_mask]
            stable_p = p_target[stable_mask]

            summary.update({
                "Recommended Depth mean": float(np.mean(stable_depth)),
                "Recommended Depth SD": float(np.std(stable_depth)),
                "Recommended Depth min": float(np.min(stable_depth)),
                "Recommended Depth max": float(np.max(stable_depth)),
                "Recommended LLBCE mean": float(np.mean(stable_llbce)),
                "Recommended LLBCE SD": float(np.std(stable_llbce)),
                "Recommended LLBCE min": float(np.min(stable_llbce)),
                "Recommended LLBCE max": float(np.max(stable_llbce)),
                "Mean P(target) in stable region": float(np.mean(stable_p)),
                "Min P(target) in stable region": float(np.min(stable_p)),
                "Max P(target) in stable region": float(np.max(stable_p)),
            })
        else:
            summary.update({
                "Recommended Depth mean": np.nan,
                "Recommended Depth SD": np.nan,
                "Recommended Depth min": np.nan,
                "Recommended Depth max": np.nan,
                "Recommended LLBCE mean": np.nan,
                "Recommended LLBCE SD": np.nan,
                "Recommended LLBCE min": np.nan,
                "Recommended LLBCE max": np.nan,
                "Mean P(target) in stable region": np.nan,
                "Min P(target) in stable region": np.nan,
                "Max P(target) in stable region": np.nan,
            })

        return summary


# ============================================================
# 6. Plotting
# ============================================================

def plot_3d_surfaces(results, class_names):
    n_classes = len(results)
    fig = plt.figure(figsize=(6.2 * n_classes, 5.4))

    for i, result in enumerate(results):
        class_label = result["class_label"]
        class_idx = result["class_idx"]
        depth_grid = result["depth_grid"]
        llbce_grid = result["llbce_grid"]
        p_target = result["p_target"]
        stable_mask = result["stable_mask"]
        best_pos = result["best_pos"]

        ax = fig.add_subplot(1, n_classes, i + 1, projection="3d")
        surf = ax.plot_surface(
            depth_grid,
            llbce_grid,
            p_target,
            cmap="RdYlBu_r",
            linewidth=0,
            antialiased=True,
            alpha=0.88,
        )

        if result["success"] and np.any(stable_mask):
            ax.scatter(
                depth_grid[stable_mask],
                llbce_grid[stable_mask],
                p_target[stable_mask],
                s=8,
                color="red",
                alpha=0.45,
                label="Recommended region",
            )

        ax.scatter(
            depth_grid[best_pos],
            llbce_grid[best_pos],
            p_target[best_pos],
            s=110,
            color="yellow",
            edgecolor="black",
            marker="*",
            label="Optimal point",
        )

        title_suffix = f"Level {result['level']}" if result["success"] else "No robust Level 1/2"
        ax.set_title(f"Type {class_label}\n{title_suffix}", fontweight="bold", pad=10)
        ax.set_xlabel("Depth of A", labelpad=9)
        ax.set_ylabel("LLBCE", labelpad=9)
        ax.set_zlabel(f"P(Type {class_label})", labelpad=9)
        ax.set_zlim(0, 1)
        ax.view_init(elev=28, azim=45)
        ax.legend(loc="upper left", fontsize=8)

        fig.colorbar(
            surf,
            ax=ax,
            shrink=0.62,
            aspect=18,
            pad=0.08,
            label=f"P(Type {class_label})",
        )

    plt.tight_layout()
    return fig


def render_recommendation_cards(summary_df):
    cols = st.columns(len(summary_df))

    for _, row in summary_df.iterrows():
        class_label = str(row["Type"])
        idx = int(class_label.replace("Type", "").strip()) - 1 if class_label.replace("Type", "").strip().isdigit() else 0
        with cols[min(max(idx, 0), len(cols) - 1)]:
            st.markdown(f"### {class_label}")
            if bool(row["Robust recommendation"]):
                st.success(str(row["Selected level"]) + " recommendation")
                st.metric("Max P(target)", f"{row['Max P(target)']:.3f}")
                st.write(f"**Optimal point**")
                st.write(f"Depth of A = **{row['Optimal Depth of A']:.2f} mm**")
                st.write(f"LLBCE = **{row['Optimal LLBCE']:.2f} mm**")
                st.write("**Recommended values, Mean ± SD**")
                st.write(
                    f"Depth of A = **{row['Recommended Depth mean']:.2f} ± {row['Recommended Depth SD']:.2f} mm**"
                )
                st.write(
                    f"LLBCE = **{row['Recommended LLBCE mean']:.2f} ± {row['Recommended LLBCE SD']:.2f} mm**"
                )
                st.write("**Recommended range**")
                st.write(
                    f"Depth of A: {row['Recommended Depth min']:.2f}–{row['Recommended Depth max']:.2f} mm"
                )
                st.write(
                    f"LLBCE: {row['Recommended LLBCE min']:.2f}–{row['Recommended LLBCE max']:.2f} mm"
                )
                st.caption(
                    f"Stable region: {int(row['Stable points'])} grid points "
                    f"({row['Stable area, %']:.2f}%)."
                )
            else:
                st.warning("No robust Level 1/2 recommendation")
                st.metric("Max P(target)", f"{row['Max P(target)']:.3f}")
                st.write("The target type was not stably supported under Level 1/2 strict criteria.")

            prob_cols = [c for c in summary_df.columns if c.startswith("P(Type")]
            prob_text = " | ".join([f"{c.replace(' at optimal','')}: {row[c]:.3f}" for c in prob_cols])
            st.caption("Probabilities at optimal point: " + prob_text)


# ============================================================
# 7. Streamlit UI
# ============================================================

st.title("🦷 BSSO Lingual Fracture-Line Prediction and Reverse Planning System")
st.markdown(
    "This system fixes patient-specific **PMBT** and **MRT**, maps the probability response "
    "surface over **Depth of A** and **LLBCE**, and outputs robust Level 1/2 model-derived "
    "candidate planning values for each lingual fracture-line type."
)

with st.sidebar:
    st.header("Input anatomical parameters")
    pmbt_input = st.number_input("PMBT", min_value=0.0, max_value=20.0, value=3.20, step=0.05, format="%.2f")
    mrt_input = st.number_input("MRT", min_value=0.0, max_value=30.0, value=9.50, step=0.05, format="%.2f")

    st.header("Computation settings")
    grid_size = st.slider("Grid density", min_value=60, max_value=200, value=120, step=10)

    st.caption("The empirical domains are pre-computed and built into the app. No raw training dataset is required.")

    with st.expander("Built-in empirical domains", expanded=False):
        for cls, domains in CLASS_DOMAINS.items():
            st.markdown(f"**Type {cls}**")
            st.write(
                f"Level 1 Depth: {domains['level1']['depth_range'][0]:.3f}–{domains['level1']['depth_range'][1]:.3f}; "
                f"LLBCE: {domains['level1']['llbce_range'][0]:.3f}–{domains['level1']['llbce_range'][1]:.3f}"
            )
            st.write(
                f"Level 2 Depth: {domains['level2']['depth_range'][0]:.3f}–{domains['level2']['depth_range'][1]:.3f}; "
                f"LLBCE: {domains['level2']['llbce_range'][0]:.3f}–{domains['level2']['llbce_range'][1]:.3f}"
            )

try:
    model, scaler, label_encoder, class_names, feature_names = load_artifacts(
        MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH
    )
    st.sidebar.success("Model artifacts loaded")
except Exception as e:
    st.error(
        "Failed to load model artifacts. Please make sure the following files exist: "
        "result/best_model.pkl, result/scaler.pkl, result/label_encoder.pkl."
    )
    st.exception(e)
    st.stop()

missing_domains = [c for c in class_names if c not in CLASS_DOMAINS]
if missing_domains:
    st.error(f"Missing hard-coded empirical domains for class(es): {missing_domains}")
    st.stop()

if st.button("Run prediction and reverse planning", type="primary", use_container_width=True):
    with st.spinner("Computing probability response surfaces and recommendations..."):
        planner = BSSOReversePlanner(
            model=model,
            scaler=scaler,
            class_names=class_names,
            feature_names=feature_names,
            grid_size=grid_size,
        )

        selected_results = []
        attempt_records = []
        summaries = []

        for cls in class_names:
            selected, attempts = planner.robust_search_for_class(
                class_label=cls,
                mrt_value=mrt_input,
                pmbt_value=pmbt_input,
            )
            selected_results.append(selected)
            summaries.append(planner.summarize_result(selected))
            for a in attempts:
                attempt_records.append({
                    "Type": f"Type {cls}",
                    "Level": a["level"],
                    "Domain": a["domain_name"],
                    "Robust recommendation": a["success"],
                    "Reason": a["reason"],
                    "Max P(target)": a["max_probability"],
                })

        summary_df = pd.DataFrame(summaries)
        attempt_df = pd.DataFrame(attempt_records)

    st.subheader("3D probability response surfaces")
    fig = plot_3d_surfaces(selected_results, class_names)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Recommended planning values and probabilities")
    render_recommendation_cards(summary_df)

    st.markdown("### Summary table")
    display_cols = [
        "Type",
        "Selected level",
        "Robust recommendation",
        "Max P(target)",
        "Optimal Depth of A",
        "Optimal LLBCE",
        "Recommended Depth mean",
        "Recommended Depth SD",
        "Recommended LLBCE mean",
        "Recommended LLBCE SD",
        "Stable area, %",
    ]
    display_df = summary_df[display_cols].copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].round(3)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download recommendation table as CSV",
        data=csv,
        file_name="bsso_reverse_planning_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Search-level log", expanded=False):
        st.dataframe(attempt_df, use_container_width=True, hide_index=True)

else:
    st.info("Enter PMBT and MRT in the sidebar, then click **Run prediction and reverse planning**.")

st.markdown("---")
st.caption(
    "Research-use decision-support prototype. Outputs are model-derived candidate planning values, "
    "not deterministic surgical instructions. Only robust Level 1/2 outputs should be considered guide-design candidates."
)