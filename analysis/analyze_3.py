#!/usr/bin/env python3
"""
Statistical analysis of multi-stage ablation results (Stages 1-4).

This is a corrected version of analyze_2.py. Differences from the original:

  1. Cohen's d is computed with the standard pooled standard deviation
     formula using unbiased variance (ddof=1):

         s_pooled = sqrt( ((n_a - 1) * s_a^2 + (n_b - 1) * s_b^2)
                          / (n_a + n_b - 2) )
         d = (mean_variant - mean_baseline) / s_pooled

     The original used sqrt((var_a + var_b) / 2) with biased variance.

  2. A 95% bootstrap percentile confidence interval is reported for d, so
     readers can see the uncertainty around the point estimate.

  3. The convergence threshold matches the paper text:

         threshold = 0.90 * max(final_best of baseline runs)

     The original used max over all stages in the group, which makes the
     metric self-referential when the winning variant defines the bar.

  4. Best-so-far curves are forward-filled per run before averaging.
     If a run did not complete the full evaluation budget, its last
     observed best_so_far value is carried forward. This prevents
     survivorship bias in the mean and CI curves.

  5. Holm-Bonferroni correction is applied to Mann-Whitney p-values
     (same as the paper text). Welch's t-test results are no longer
     emitted, so there is no ambiguity about which p-values are
     corrected.

  6. Each output row records n_runs explicitly for transparency.

  7. cummax is applied post-load as a safety net so that best_so_far is
     guaranteed monotonic per run, regardless of how the CSV was written.

Input:  trials.csv per stage, with columns trial, run_id, fitness or
        best_so_far. If only fitness is available, best_so_far is derived
        as a per-run running maximum.

Output: per-group directory under --out containing:
        - evolution_ci.png         (mean +/- 95% CI of best_so_far)
        - final_boxplot.png        (final_best distribution per stage)
        - convergence_hist.png     (trials-to-threshold histogram)
        - final_summary.csv        (mean, std, n_runs per stage)
        - convergence_summary.csv  (mean, std of trials-to-threshold)
        - stability_summary.csv    (variance, CV on tail of each run)
        - stats_vs_baseline.csv    (Mann-Whitney + Holm + Cohen's d + CI)
        - summary.json             (config snapshot for the group)

Usage:
    python analyze_3.py --out results --auto_discover
    python analyze_3.py --out results --groups perception method
    python analyze_3.py --out results --groups perception \\
        --baseline perception:stage1b_pretrained
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SQUARE_FIGSIZE = (3.5, 3.5)

#Plot style
def set_ieee_style(single_column: bool = True):
    width_in = 3.5 if single_column else 5.5
    height_in = width_in
    mpl.rcParams.update({
        "figure.figsize": (width_in, height_in),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
    })



# Stages / labels / groups
STAGES_DEFAULT = {
    # Stage 1: Perception ablation
    "stage1a_random": "logs/runs_ga_stage1_randomcnn/trials.csv",
    "stage1b_pretrained": "logs/runs_ga_stage1_pretrainedcnn/trials.csv",

    # Stage 2: Evolution with adapters
    "stage2a": "logs/runs_ga_stage2a_evolve_cnn_only/trials.csv",
    "stage2b": "logs/runs_ga_stage2b_coevo/trials.csv",

    # Stage 3: Hybrid
    "stage3_hybrid": "logs/runs_ga_hybrid_minimal/trials.csv",

    # Stage 4: Stabilization mechanisms
    "stage4a_Stabilized_hybrid": "logs/runs_ga_stage4a_hybrid_hof_trimmed/trials.csv",
    "stage4b_Adaptive_hybrid": "logs/runs_ga_stage4b_hybrid_adaptive/trials.csv",

    # Other comparisons
    "warehouse_bo_vision8":
        "logs/logs-warehouse/runs_bo_vision8/warehouse_bo_trials.csv",
    "warehouse_ga_vision8":
        "logs/logs-warehouse/runs_ga_vision8/warehouse_ga_trials.csv",
}

LABELS_DEFAULT = {
    "stage1a_random": "Frozen Random CNN (evolve bp)",
    "stage1b_pretrained": "Frozen Pretrained CNN (evolve bp)",
    "stage2a": "Evolve CNN Only (evolve adapters)",
    "stage2b": "Co-evolution (evolve adapters + bp)",
    "stage3_hybrid": "Minimal Hybrid (evolve CNN + bp)",
    "stage4a_Stabilized_hybrid": "Minimal Hybrid + Trimmed Mean + HoF",
    "stage4b_Adaptive_hybrid": "Minimal Hybrid + Trimmed Mean + HoF + Adaptive Mutation",
    "warehouse_bo_vision8": "BO",
    "warehouse_ga_vision8": "GA",
}

GROUPS_DEFAULT = {
    "perception": [
        "stage1a_random",
        "stage1b_pretrained",
    ],
    "method": [
        "stage2a",
        "stage2b",
        "stage3_hybrid",
    ],
    "stage2_adapters": [
        "stage2a",
        "stage2b",
    ],
    "mechanisms": [
        "stage3_hybrid",
        "stage4a_Stabilized_hybrid",
        "stage4b_Adaptive_hybrid",
    ],
    "stage3_main": [
        "stage1b_pretrained",
        "stage2b",
        "stage3_hybrid",
    ],
    "warehouse_bo_vs_ga": [
        "warehouse_ga_vision8",
        "warehouse_bo_vision8",
    ],
    "all": [],  # filled automatically
}


# CSV loading and per-run curve extraction
def load_csv(path: str) -> pd.DataFrame:
    """Load a trials.csv and guarantee a monotonic best_so_far per run."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for c in ["trial", "run_id", "fitness", "best_so_far"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "trial" not in df.columns:
        raise ValueError(f"{path}: missing 'trial' column.")
    if "run_id" not in df.columns:
        df["run_id"] = 0

    # Derive best_so_far if missing.
    if "best_so_far" not in df.columns:
        if "fitness" not in df.columns:
            raise ValueError(
                f"{path}: missing both 'best_so_far' and 'fitness' columns."
            )
        df = df.sort_values(["run_id", "trial"]).reset_index(drop=True)
        df["best_so_far"] = df.groupby("run_id")["fitness"].cummax()
    else:
        # Safety net: enforce monotonicity even if best_so_far is in the
        # CSV. Some logs may write non-monotonic rows by accident.
        df = df.sort_values(["run_id", "trial"]).reset_index(drop=True)
        df["best_so_far"] = df.groupby("run_id")["best_so_far"].cummax()

    return df


def per_run_best_curve(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Return {run_id: DataFrame[trial, best_so_far]} sorted by trial."""
    out = {}
    for rid, g in df.groupby("run_id"):
        gg = g.sort_values("trial")[["trial", "best_so_far"]].dropna()
        out[int(rid)] = gg.reset_index(drop=True)
    return out


def align_and_average(
    curves: Dict[int, pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Forward-fill each run to the global max trial, then average.

    For each run we extend the best_so_far value at the run's final trial
    forward up to the global max trial across all runs. This eliminates
    survivorship bias: at any trial index, the mean and CI are computed
    over the same number of runs.
    """
    if not curves:
        return None, None

    # Determine the maximum trial index across all runs.
    global_tmax = max(int(c["trial"].max()) for c in curves.values() if not c.empty)
    trials_axis = np.arange(1, global_tmax + 1, dtype=int)

    padded_rows = []
    for rid, c in curves.items():
        if c.empty:
            continue
        # Reindex onto trials_axis. Missing values come from gaps in the log;
        # also from runs that stopped before global_tmax. We forward-fill
        # both cases, then back-fill any leading NaN with the first value.
        s = c.set_index("trial")["best_so_far"].reindex(trials_axis)
        s = s.ffill().bfill()
        for t, val in zip(trials_axis, s.values):
            padded_rows.append({"trial": int(t), "run_id": int(rid),
                                "best_so_far": float(val)})

    M = pd.DataFrame(padded_rows)
    agg = (
        M.groupby("trial")["best_so_far"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["stderr"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    return M, agg


# Per-run aggregates
def final_best_by_run(curves: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """For each run, return the final best_so_far value (last trial)."""
    rows = []
    for rid, c in curves.items():
        if c.empty:
            rows.append({"run_id": rid, "final_best": np.nan,
                         "final_trial": np.nan})
            continue
        # best_so_far is guaranteed monotonic, so the last row is the max.
        last = c.iloc[-1]
        rows.append({
            "run_id": rid,
            "final_best": float(last["best_so_far"]),
            "final_trial": int(last["trial"]),
        })
    return pd.DataFrame(rows)


def trials_to_threshold(
    curves: Dict[int, pd.DataFrame],
    threshold: float,
) -> pd.DataFrame:
    """For each run, find the earliest trial where best_so_far >= threshold."""
    rows = []
    for rid, c in curves.items():
        if c.empty or np.isnan(threshold):
            rows.append({"run_id": rid, "trials_to_threshold": np.nan})
            continue
        hit = c[c["best_so_far"] >= threshold]
        t = int(hit["trial"].iloc[0]) if not hit.empty else np.nan
        rows.append({"run_id": rid, "trials_to_threshold": t})
    return pd.DataFrame(rows)


def stability_metrics(
    curves: Dict[int, pd.DataFrame],
    tail_pct: float = 0.20,
) -> pd.DataFrame:
    """Variance and coefficient of variation in the tail of each run.

    The tail is the last `tail_pct` fraction of trials. With monotonic
    best_so_far, the tail variance reflects how late improvements arrived
    rather than within-run jitter, but it is still a useful comparator
    across stages.
    """
    rows = []
    for rid, c in curves.items():
        if c.empty:
            rows.append({"run_id": rid, "variance": np.nan, "cv": np.nan})
            continue
        tmax = c["trial"].max()
        cut = tmax - tail_pct * max(1, tmax)
        tail = c[c["trial"] >= cut]["best_so_far"]
        if len(tail) > 1:
            var = float(np.var(tail, ddof=1))
            mean = float(np.mean(tail))
            # Guard against near-zero means so CV does not explode.
            cv = float(np.std(tail, ddof=1) / mean) if abs(mean) > 1e-9 else np.nan
        else:
            var, cv = np.nan, np.nan
        rows.append({"run_id": rid, "variance": var, "cv": cv})
    return pd.DataFrame(rows)


# Effect size
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Cohen's d for two independent samples.

    Uses unbiased variance (ddof=1) and the canonical pooled standard
    deviation:

        s_pooled = sqrt( ((n_a - 1) s_a^2 + (n_b - 1) s_b^2)
                         / (n_a + n_b - 2) )

    The sign is (mean_b - mean_a) / s_pooled, so d > 0 means b > a.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / np.sqrt(pooled_var))


def cohens_d_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
    rng_seed: int = 12345,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for Cohen's d.

    Resamples a and b independently with replacement, recomputes d each
    time, and returns the (alpha/2, 1 - alpha/2) percentiles. With small
    sample sizes the d point estimate has wide uncertainty, and showing
    the CI alongside d is the honest way to report it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    n_a, n_b = len(a), len(b)
    ds = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        a_s = rng.choice(a, size=n_a, replace=True)
        b_s = rng.choice(b, size=n_b, replace=True)
        ds[i] = cohens_d(a_s, b_s)
    ds = ds[~np.isnan(ds)]
    if ds.size == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - ci
    lo = float(np.percentile(ds, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(ds, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


# Multiple comparisons correction
def holm_correction(pvals: List[float]) -> List[float]:
    pvals_arr = np.array(pvals, dtype=float)
    m = len(pvals_arr)
    if m == 0:
        return []

    order = np.argsort(pvals_arr)
    sorted_p = pvals_arr[order]

    sorted_adj = np.empty(m, dtype=float)
    for i, p in enumerate(sorted_p):
        sorted_adj[i] = min(1.0, (m - i) * p)

    # Holm adjusted p-values must be non-decreasing in sorted order.
    sorted_adj = np.maximum.accumulate(sorted_adj)
    sorted_adj = np.minimum(sorted_adj, 1.0)

    adj = np.empty(m, dtype=float)
    adj[order] = sorted_adj
    return adj.tolist()


# Plots
def plot_group_evolution_with_ci(group_name, stage_dfs, labels, out_path):
    plt.figure(figsize=SQUARE_FIGSIZE)
    for key, df in stage_dfs.items():
        curves = per_run_best_curve(df)
        _, agg = align_and_average(curves)
        if agg is None:
            continue
        plt.plot(agg["trial"], agg["mean"], label=labels.get(key, key))
        plt.fill_between(
            agg["trial"],
            agg["mean"] - 1.96 * agg["stderr"],
            agg["mean"] + 1.96 * agg["stderr"],
            alpha=0.2,
        )
    plt.xlabel("Trial")
    plt.ylabel("Best-so-far fitness (mean ± 95% CI)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_group_final_boxplot(group_name, finals_by_stage, labels, out_path):
    def _wrap_label(s, max_len=18):
        s = str(s)
        if len(s) <= max_len:
            return s
        parts = s.split(" ")
        lines, cur = [], ""
        for w in parts:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_len:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) > 2:
            lines = [lines[0], " ".join(lines[1:])]
        return "\n".join(lines)

    plt.figure(figsize=SQUARE_FIGSIZE)
    keys = list(finals_by_stage.keys())
    data = [finals_by_stage[k] for k in keys]
    tick = [_wrap_label(labels.get(k, k)) for k in keys]
    plt.boxplot(data, tick_labels=tick, patch_artist=True,
                boxprops=dict(alpha=0.6))
    plt.ylabel("Final best-so-far fitness")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.gcf().subplots_adjust(bottom=0.22)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_group_convergence_hist(group_name, conv_by_stage, labels, out_path):
    plt.figure(figsize=SQUARE_FIGSIZE)

    cleaned = {
        k: np.asarray(vals, dtype=float)[~np.isnan(vals)]
        for k, vals in conv_by_stage.items()
    }

    pooled = np.concatenate([v for v in cleaned.values() if v.size > 0]) \
        if any(v.size > 0 for v in cleaned.values()) else np.array([])

    if pooled.size == 0:
        plt.text(0.5, 0.5, "No convergence data",
                 ha="center", va="center")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return

    # Trials are discrete integer values, so use grouped bars instead of
    # overlapping histograms.
    x_vals = np.arange(int(np.nanmin(pooled)), int(np.nanmax(pooled)) + 1)
    stage_keys = list(cleaned.keys())
    n_stages = len(stage_keys)

    total_width = 0.75
    bar_width = total_width / max(1, n_stages)

    for i, k in enumerate(stage_keys):
        v = cleaned[k]
        if v.size == 0:
            continue

        counts = np.array([np.sum(v == x) for x in x_vals], dtype=int)

        offset = (i - (n_stages - 1) / 2) * bar_width

        plt.bar(
            x_vals + offset,
            counts,
            width=bar_width,
            label=labels.get(k, k),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    plt.xlabel("Trials to threshold")
    plt.ylabel("Frequency")
    plt.xticks(x_vals)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# Per-group analysis
def analyze_group(
    group_name: str,
    stage_keys: List[str],
    all_data: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    out_dir: str,
    baseline_key: Optional[str] = None,
    pct_of_baseline: float = 0.90,
    tail_pct: float = 0.20,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 12345,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_dfs = {k: all_data[k] for k in stage_keys if k in all_data}
    if len(stage_dfs) < 1:
        print(f"[WARN] Group '{group_name}': no data. Skipping.")
        return

    # Baseline selection: explicit override if given, else first stage in
    # the configured order. We never search for a magical "baseline" key.
    if baseline_key is None:
        baseline_key = list(stage_dfs.keys())[0]
        print(f"[INFO] Group '{group_name}': no baseline specified, "
              f"using first stage '{baseline_key}'.")
    elif baseline_key not in stage_dfs:
        print(f"[WARN] Group '{group_name}': baseline '{baseline_key}' "
              f"not found. Falling back to first stage.")
        baseline_key = list(stage_dfs.keys())[0]

    # --- 1) Evolution CI plot ---
    plot_group_evolution_with_ci(
        group_name, stage_dfs, labels,
        str(out_dir / f"{group_name}_evolution_ci.png"),
    )

    # --- 2) Final values per run, per stage ---
    finals_by_stage: Dict[str, np.ndarray] = {}
    final_rows = []
    for k, df in stage_dfs.items():
        curves = per_run_best_curve(df)
        fb = final_best_by_run(curves)["final_best"].dropna().values
        finals_by_stage[k] = fb
        final_rows.append({
            "group": group_name,
            "stage": k,
            "label": labels.get(k, k),
            "n_runs": int(len(fb)),
            "final_best_mean": float(np.mean(fb)) if len(fb) else np.nan,
            "final_best_std": float(np.std(fb, ddof=1))
                              if len(fb) > 1 else np.nan,
            "final_best_min": float(np.min(fb)) if len(fb) else np.nan,
            "final_best_max": float(np.max(fb)) if len(fb) else np.nan,
        })
    pd.DataFrame(final_rows).to_csv(
        out_dir / f"{group_name}_final_summary.csv", index=False,
    )
    plot_group_final_boxplot(
        group_name, finals_by_stage, labels,
        str(out_dir / f"{group_name}_final_boxplot.png"),
    )

    # --- 3) Convergence threshold from BASELINE only ---
    # threshold = pct_of_baseline * max(baseline final_best values)
    # This matches the paper text: "theta = 0.90 * max(final_best) of baseline".
    baseline_final = finals_by_stage.get(baseline_key, np.array([]))
    if baseline_final.size > 0:
        threshold = float(pct_of_baseline) * float(np.nanmax(baseline_final))
    else:
        threshold = float("nan")
        print(f"[WARN] Group '{group_name}': baseline has no final values; "
              f"convergence threshold is NaN.")

    conv_by_stage: Dict[str, np.ndarray] = {}
    conv_rows = []
    for k, df in stage_dfs.items():
        curves = per_run_best_curve(df)
        conv = trials_to_threshold(curves, threshold)["trials_to_threshold"].values
        conv_by_stage[k] = conv
        n_reached = int(np.sum(~np.isnan(conv)))
        conv_rows.append({
            "group": group_name,
            "stage": k,
            "label": labels.get(k, k),
            "threshold": threshold,
            "threshold_basis": f"{pct_of_baseline:.2f}*max(baseline)",
            "baseline_stage": baseline_key,
            "n_runs": int(len(conv)),
            "n_runs_reached": n_reached,
            "mean_trials_to_threshold": float(np.nanmean(conv))
                                        if n_reached > 0 else np.nan,
            "std_trials_to_threshold": float(np.nanstd(conv, ddof=1))
                                       if n_reached > 1 else np.nan,
        })
    pd.DataFrame(conv_rows).to_csv(
        out_dir / f"{group_name}_convergence_summary.csv", index=False,
    )
    plot_group_convergence_hist(
        group_name, conv_by_stage, labels,
        str(out_dir / f"{group_name}_convergence_hist.png"),
    )

    # --- 4) Stability (tail variance / CV) ---
    stab_rows = []
    for k, df in stage_dfs.items():
        curves = per_run_best_curve(df)
        stab = stability_metrics(curves, tail_pct=tail_pct)
        stab_rows.append({
            "group": group_name,
            "stage": k,
            "label": labels.get(k, k),
            "tail_pct": tail_pct,
            "n_runs": int(len(stab)),
            "mean_variance": float(np.nanmean(stab["variance"].values))
                             if not stab["variance"].isna().all() else np.nan,
            "mean_cv": float(np.nanmean(stab["cv"].values))
                       if not stab["cv"].isna().all() else np.nan,
        })
    pd.DataFrame(stab_rows).to_csv(
        out_dir / f"{group_name}_stability_summary.csv", index=False,
    )

    # --- 5) Statistical tests vs baseline ---
    # Mann-Whitney U, two-sided, Holm-Bonferroni corrected.
    # Cohen's d with bootstrap 95% CI for effect size.
    stat_rows = []
    pvals: List[float] = []
    row_idx_for_pvals: List[int] = []

    for k in stage_dfs.keys():
        if k == baseline_key:
            continue
        b = baseline_final
        v = finals_by_stage.get(k, np.array([]))

        if len(b) < 2 or len(v) < 2:
            p_mw = float("nan")
            d = float("nan")
            d_lo, d_hi = float("nan"), float("nan")
        else:
            p_mw = float(
                stats.mannwhitneyu(b, v, alternative="two-sided").pvalue
            )
            d = cohens_d(b, v)
            d_lo, d_hi = cohens_d_bootstrap_ci(
                b, v,
                n_resamples=bootstrap_resamples,
                ci=0.95,
                rng_seed=bootstrap_seed,
            )

        pct_improvement = (
            float(((np.mean(v) - np.mean(b)) / abs(np.mean(b))) * 100.0)
            if len(b) and len(v) and abs(np.mean(b)) > 1e-9
            else float("nan")
        )

        row = {
            "group": group_name,
            "baseline_stage": baseline_key,
            "baseline_label": labels.get(baseline_key, baseline_key),
            "variant_stage": k,
            "variant_label": labels.get(k, k),
            "metric": "final_best",
            "n_baseline": int(len(b)),
            "n_variant": int(len(v)),
            "baseline_mean": float(np.mean(b)) if len(b) else np.nan,
            "variant_mean": float(np.mean(v)) if len(v) else np.nan,
            "baseline_std": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
            "variant_std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
            "p_value_mannwhitney_raw": p_mw,
            "cohens_d": d,
            "cohens_d_ci_lo_95": d_lo,
            "cohens_d_ci_hi_95": d_hi,
            "pct_improvement": pct_improvement,
        }
        stat_rows.append(row)
        if not np.isnan(p_mw):
            pvals.append(p_mw)
            row_idx_for_pvals.append(len(stat_rows) - 1)

    if pvals:
        adj = holm_correction(pvals)
        for j, idx in enumerate(row_idx_for_pvals):
            stat_rows[idx]["p_value_mannwhitney_holm"] = float(adj[j])
            stat_rows[idx]["significant_0.05_holm"] = bool(adj[j] < 0.05)
            stat_rows[idx]["significant_0.01_holm"] = bool(adj[j] < 0.01)

    pd.DataFrame(stat_rows).to_csv(
        out_dir / f"{group_name}_stats_vs_baseline.csv", index=False,
    )

    # --- 6) Group summary JSON ---
    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "group": group_name,
        "stages_included": list(stage_dfs.keys()),
        "baseline": baseline_key,
        "baseline_label": labels.get(baseline_key, baseline_key),
        "convergence_threshold_value": threshold,
        "convergence_threshold_rule": (
            f"{pct_of_baseline} * max(final_best) of baseline runs"
        ),
        "tail_pct_for_stability": tail_pct,
        "statistical_test": "Mann-Whitney U, two-sided",
        "multiple_comparisons_correction": "Holm-Bonferroni",
        "effect_size": "Cohen's d (pooled SD, ddof=1)",
        "effect_size_uncertainty": (
            f"Bootstrap percentile 95% CI, {bootstrap_resamples} resamples"
        ),
        "ci_curves": "Forward-filled per run; mean +/- 1.96 * SE",
    }
    with open(out_dir / f"{group_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Group '{group_name}' done -> {out_dir}")


# Auto-discover
def autodiscover_trials(logs_root: str):
    root = Path(logs_root)
    found = list(root.rglob("trials.csv"))
    stages = {}
    labels = {}
    for p in found:
        stage_key = p.parent.name
        key = stage_key
        i = 2
        while key in stages:
            key = f"{stage_key}_{i}"
            i += 1
        stages[key] = str(p)
        labels[key] = stage_key
    return stages, labels


def parse_baseline_overrides(items):
    out = {}
    if not items:
        return out
    for it in items:
        if ":" not in it:
            continue
        g, s = it.split(":", 1)
        out[g.strip()] = s.strip()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Statistical analysis of Stages 1-4 ablation results. "
            "Stage 5 (paired fitness-mode comparison) is handled by a "
            "separate script and is not analyzed here."
        )
    )
    ap.add_argument("--out", default="results",
                    help="Output directory")
    ap.add_argument("--groups", nargs="*", default=["all"],
                    help="Groups to analyze (default: all)")
    ap.add_argument("--baseline", action="append", default=[],
                    help="Per-group baseline override: group:stage_key")
    ap.add_argument("--pct_of_baseline", type=float, default=0.90,
                    help=("Convergence threshold = pct_of_baseline * "
                          "max(final_best of baseline runs)"))
    ap.add_argument("--tail_pct", type=float, default=0.20,
                    help="Tail fraction for stability metrics")
    ap.add_argument("--bootstrap_resamples", type=int, default=10000,
                    help="Bootstrap resamples for Cohen's d CI")
    ap.add_argument("--bootstrap_seed", type=int, default=12345,
                    help="RNG seed for bootstrap reproducibility")
    ap.add_argument("--auto_discover", action="store_true",
                    help="Auto-discover trials.csv under --logs_root")
    ap.add_argument("--logs_root", default="logs",
                    help="Root directory for logs")
    ap.add_argument("--single_column", action="store_true",
                    help="IEEE single-column figure size")
    args = ap.parse_args()

    set_ieee_style(single_column=bool(args.single_column))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.auto_discover:
        stages, labels = autodiscover_trials(args.logs_root)
        if not stages:
            raise RuntimeError(f"No trials.csv under: {args.logs_root}")
    else:
        stages = dict(STAGES_DEFAULT)
        labels = dict(LABELS_DEFAULT)

    groups = dict(GROUPS_DEFAULT)
    groups["all"] = list(stages.keys())

    all_data = {}
    for key, path in stages.items():
        if os.path.exists(path):
            try:
                df = load_csv(path)
                df["stage"] = key
                all_data[key] = df
            except Exception as e:
                print(f"[WARN] Failed loading {path}: {e}")
        else:
            print(f"[WARN] Missing: {path}")

    baseline_overrides = parse_baseline_overrides(args.baseline)

    cfg = {
        "out": str(out_root),
        "auto_discover": bool(args.auto_discover),
        "logs_root": args.logs_root,
        "groups_requested": args.groups,
        "stages_loaded": {k: stages[k] for k in all_data.keys()},
        "labels": {k: labels.get(k, k) for k in all_data.keys()},
        "groups_effective": {g: groups.get(g, []) for g in args.groups},
        "baseline_overrides": baseline_overrides,
        "pct_of_baseline": args.pct_of_baseline,
        "tail_pct": args.tail_pct,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "convergence_rule": (
            f"threshold = {args.pct_of_baseline} * "
            f"max(final_best of baseline runs)"
        ),
        "statistical_test": "Mann-Whitney U, two-sided",
        "multiple_comparisons_correction": "Holm-Bonferroni",
        "effect_size": "Cohen's d with bootstrap 95% CI",
        "padding_policy": (
            "Per-run best_so_far is forward-filled to the global max "
            "trial index before averaging, to avoid survivorship bias."
        ),
    }
    with open(out_root / "analysis_config_snapshot.json", "w") as f:
        json.dump(cfg, f, indent=2)

    for g in args.groups:
        if g not in groups:
            print(f"[WARN] Unknown group '{g}'. "
                  f"Available: {list(groups.keys())}")
            continue
        stage_keys = list(all_data.keys()) if g == "all" else groups[g]
        analyze_group(
            group_name=g,
            stage_keys=stage_keys,
            all_data=all_data,
            labels=labels,
            out_dir=str(out_root / g),
            baseline_key=baseline_overrides.get(g, None),
            pct_of_baseline=args.pct_of_baseline,
            tail_pct=args.tail_pct,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )

    print(f"\n[DONE] Outputs in: {out_root}/")


if __name__ == "__main__":
    main()
