"""Statistical comparison of two optimizer runs (GA vs BO).

Compares two trial-log CSVs — a baseline and a variant — and produces
convergence, stability, and statistical comparison outputs.

Input CSVs are expected to have columns: trial, run_id, fitness,
best_so_far, and optionally params_json (for diversity analysis).

Usage:
    python BOvsGA_analyze.py --baseline ga_trials.csv --variant bo_trials.csv \\
        --blabel GA --vlabel BO --out analysis_out/

Outputs (written to --out):
    evolution_with_ci.png       Mean best-so-far curves with 95% CI
    final_best_boxplot.png      Distribution of final best fitness per run
    convergence_histogram.png   Trials-to-threshold distribution
    param_diversity.png         Parameter diversity over trials (if params_json present)
    convergence.csv              Per-run trials-to-threshold
    stability.csv                Per-run variance/CV over the final tail of trials
    statistical_tests.csv        t-test, Mann-Whitney U, Cohen's d per metric
    summary.json                 Full numeric summary
    comparison_table.tex         LaTeX table

Methodology notes:
    - Convergence threshold: --target sets an absolute fitness value;
      otherwise --pct_of_best (default 0.90) × the global best final
      score across both methods. If a run never reaches the threshold,
      its trials_to_threshold is NaN.
    - Statistical tests are unpaired (independent t-test, Mann-Whitney
      U) because baseline and variant runs are independent (no shared
      seed/initial point between matching run_ids).
    - Cohen's d uses sample variance (ddof=1).
"""

import argparse, os, json
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from pathlib import Path

# ============================================================================
# PLOTTING STYLE (IEEE-like)
# ============================================================================

def set_ieee_style(single_column: bool = True):
    width_in = 3.5 if single_column else 7.16
    height_in = 2.6

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


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_csv(path):
    """Load trials CSV and ensure numeric types"""
    df = pd.read_csv(path)
    for c in ["trial", "run_id", "fitness", "best_so_far"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def per_run_best_curve(df):
    """Extract best-so-far curves per run_id"""
    out = {}
    for rid, g in df.groupby("run_id"):
        gg = g.sort_values("trial")[["trial", "best_so_far"]].dropna()
        out[int(rid)] = gg
    return out

def align_and_average(curves):
    """Align curves by trial and compute mean/stderr"""
    all_df = []
    for rid, c in curves.items():
        c2 = c.copy()
        c2["run_id"] = int(rid)
        all_df.append(c2)
    if not all_df:
        return None, None
    M = pd.concat(all_df, ignore_index=True)
    agg = M.groupby("trial")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
    agg["stderr"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    return M, agg

# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

def trials_to_threshold(curves, threshold):
    """Find first trial where best_so_far >= threshold for each run"""
    rows = []
    for rid, c in curves.items():
        hit = c[c["best_so_far"] >= threshold]
        t = int(hit["trial"].iloc[0]) if not hit.empty else np.nan
        rows.append({"run_id": rid, "trials_to_threshold": t})
    return pd.DataFrame(rows)

def final_best_by_run(curves):
    """Extract final best fitness for each run"""
    rows = []
    for rid, c in curves.items():
        val = float(c["best_so_far"].max()) if not c.empty else np.nan
        rows.append({"run_id": rid, "final_best": val})
    return pd.DataFrame(rows)

# ============================================================================
# STABILITY ANALYSIS
# ============================================================================

def stability_metrics(curves, tail_pct=0.20):
    """
    Compute stability metrics over last tail_pct% of trials:
    - Variance of best_so_far
    - Coefficient of variation (std/mean)
    tail_pct: fraction of final trials considered "converged" for stability
    analysis (default 0.20, i.e. last 20% of trials).
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
            var = float(np.var(tail))
            mean = float(np.mean(tail))
            cv = float(np.std(tail) / mean) if mean != 0 else np.nan
        else:
            var, cv = np.nan, np.nan
        
        rows.append({"run_id": rid, "variance": var, "cv": cv})
    return pd.DataFrame(rows)

# ============================================================================
# DOMAIN-SPECIFIC METRICS (from params_json if available)
# ============================================================================

def extract_domain_metrics(df):
    """
    Parse params_json to extract domain-specific insights:
    - Parameter distributions
    - Gene diversity over time
    """
    if "params_json" not in df.columns:
        return None
    
    params_list = []
    for idx, row in df.iterrows():
        try:
            p = json.loads(row["params_json"])
            p["trial"] = row["trial"]
            p["run_id"] = row["run_id"]
            p["fitness"] = row["fitness"]
            params_list.append(p)
        except:
            continue
    
    if not params_list:
        return None
    
    return pd.DataFrame(params_list)



def compute_diversity(params_df, param_keys):
    """Compute population diversity per trial.

    For each trial, returns the mean standard deviation across all
    numeric parameters (param_diversity) — a proxy for exploration
    vs. exploitation over the course of the run.

    Columns holding dict/list/object values (e.g. adapters_groups) are
    skipped, as std is undefined for them.

    Returns:
        DataFrame with columns ['trial', 'param_diversity'].
    """

    if params_df is None or len(params_df) == 0:
        return pd.DataFrame(columns=["trial", "param_diversity"])

    rows = []
    grouped = params_df.groupby("trial")

    for trial, g in grouped:
        stds = []
        for k in param_keys:
            if k not in g.columns:
                continue
            s = g[k]

            first = s.iloc[0]
            if isinstance(first, (dict, list)):
                continue

            try:
                vals = s.astype(float)
            except Exception:
                continue

            std_val = float(vals.std())
            if not np.isnan(std_val):
                stds.append(std_val)

        if stds:
            rows.append({
                "trial": int(trial),
                "param_diversity": float(np.mean(stds))
            })

    if not rows:
        return pd.DataFrame(columns=["trial", "param_diversity"])

    df = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    return df


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def statistical_comparison(baseline_vals, variant_vals, metric_name="metric"):
    """Compare baseline vs variant using independent-samples tests.
    - t-test (parametric)
    - Mann-Whitney U (non-parametric)
    - Effect size (Cohen's d)

    Note: assumes baseline and variant runs are NOT paired by seed/run_id.
    If runs are paired (same seed across methods), use a paired test
    (Wilcoxon signed-rank) instead for higher statistical power.
    """
    b = np.array([x for x in baseline_vals if not np.isnan(x)])
    v = np.array([x for x in variant_vals if not np.isnan(x)])
    
    if len(b) < 2 or len(v) < 2:
        return {
            "metric": metric_name,
            "baseline_mean": float(np.mean(b)) if len(b) > 0 else np.nan,
            "variant_mean": float(np.mean(v)) if len(v) > 0 else np.nan,
            "p_value_ttest": np.nan,
            "p_value_mannwhitney": np.nan,
            "cohens_d": np.nan,
            "significant_0.05": False,
        }
    
    # t-test
    t_stat, p_ttest = stats.ttest_ind(b, v)
    
    # Mann-Whitney U test 
    u_stat, p_mw = stats.mannwhitneyu(b, v, alternative='two-sided')
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.var(b, ddof=1) + np.var(v, ddof=1)) / 2)
    cohens_d = (np.mean(v) - np.mean(b)) / pooled_std if pooled_std != 0 else np.nan
    
    # Percentage improvement
    pct_improvement = ((np.mean(v) - np.mean(b)) / abs(np.mean(b)) * 100) if np.mean(b) != 0 else np.nan
    
    return {
        "metric": metric_name,
        "baseline_mean": float(np.mean(b)),
        "baseline_std": float(np.std(b)),
        "variant_mean": float(np.mean(v)),
        "variant_std": float(np.std(v)),
        "p_value_ttest": float(p_ttest),
        "p_value_mannwhitney": float(p_mw),
        "cohens_d": float(cohens_d),
        "pct_improvement": float(pct_improvement),
        "significant_0.05": bool(p_ttest < 0.05),
        "significant_0.01": bool(p_ttest < 0.01),
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_evolution_with_ci(b_agg, v_agg, blabel, vlabel, out_path):
    """Plot mean best-so-far curve with 95% CI (mean ± 1.96·SEM, normal approximation).

    Note: with small run counts, the normal approximation may not hold,
    consider bootstrap CIs for n < 10 runs.
    """
    plt.figure()  
    
    if b_agg is not None:
        plt.plot(b_agg["trial"], b_agg["mean"], label=blabel)
        plt.fill_between(
            b_agg["trial"],
            b_agg["mean"] - 1.96 * b_agg["stderr"],
            b_agg["mean"] + 1.96 * b_agg["stderr"],
            alpha=0.2
        )
    
    if v_agg is not None:
        plt.plot(v_agg["trial"], v_agg["mean"], label=vlabel)
        plt.fill_between(
            v_agg["trial"],
            v_agg["mean"] - 1.96 * v_agg["stderr"],
            v_agg["mean"] + 1.96 * v_agg["stderr"],
            alpha=0.2
        )
    
    plt.xlabel("Trial")
    plt.ylabel("Best-so-far fitness (mean ± 95% CI)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_final_distribution(b_final, v_final, blabel, vlabel, out_path):
    """Box plot for final best distribution"""
    plt.figure()
    
    b_vals = b_final["final_best"].dropna().values
    v_vals = v_final["final_best"].dropna().values

    plt.boxplot(
        [b_vals, v_vals],
        labels=[blabel, vlabel],
        patch_artist=True,
        boxprops=dict(alpha=0.6),
        medianprops=dict(linewidth=1.2),
    )
    
    plt.ylabel("Final best-so-far fitness")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_convergence_histogram(b_conv, v_conv, blabel, vlabel, out_path):
    """Histogram of trials to reach threshold"""
    plt.figure()
    
    b_vals = b_conv["trials_to_threshold"].dropna().values
    v_vals = v_conv["trials_to_threshold"].dropna().values
    
    if len(b_vals) > 0 or len(v_vals) > 0:
        bins = np.linspace(
            min(b_vals.min() if len(b_vals) > 0 else np.inf,
                v_vals.min() if len(v_vals) > 0 else np.inf),
            max(b_vals.max() if len(b_vals) > 0 else -np.inf,
                v_vals.max() if len(v_vals) > 0 else -np.inf),
            15
        )
        
        if len(b_vals) > 0:
            plt.hist(b_vals, bins=bins, alpha=0.6, label=blabel, edgecolor="black")
        if len(v_vals) > 0:
            plt.hist(v_vals, bins=bins, alpha=0.6, label=vlabel, edgecolor="black")
    
    plt.xlabel("Trials to threshold")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Enhanced GA comparison with statistical tests")
    ap.add_argument("--baseline", required=True, help="Path to baseline trials.csv")
    ap.add_argument("--variant", required=True, help="Path to variant trials.csv")
    ap.add_argument("--blabel", default="Baseline", help="Label for baseline")
    ap.add_argument("--vlabel", default="Variant", help="Label for variant")
    ap.add_argument("--out", default="analysis_out", help="Output directory")
    ap.add_argument("--target", type=float, default=None, 
                    help="Absolute fitness target for convergence")
    ap.add_argument("--pct_of_best", type=float, default=0.90,
                    help="If no --target, use % of max(final_best)")
    args = ap.parse_args()

    # Apply IEEE-like plotting style
    set_ieee_style(single_column=True)


    os.makedirs(args.out, exist_ok=True)

    # Load data
    print(f"Loading baseline: {args.baseline}")
    b = load_csv(args.baseline)
    print(f"Loading variant: {args.variant}")
    v = load_csv(args.variant)

    b_curves = per_run_best_curve(b)
    v_curves = per_run_best_curve(v)

    _, b_agg = align_and_average(b_curves)
    _, v_agg = align_and_average(v_curves)

    # ========================================================================
    # 1) EVOLUTION PLOT
    # ========================================================================
    print("\n[1/8] Generating evolution plot with confidence intervals...")
    plot_evolution_with_ci(
        b_agg, v_agg, args.blabel, args.vlabel,
        os.path.join(args.out, "evolution_with_ci.png")
    )

    # ========================================================================
    # 2) CONVERGENCE ANALYSIS
    # ========================================================================
    print("[2/8] Computing convergence metrics...")
    
    # Determine threshold
    if args.target is not None:
        target = float(args.target)
    else:
        b_final = final_best_by_run(b_curves)
        v_final = final_best_by_run(v_curves)
        max_final = pd.concat([b_final["final_best"], v_final["final_best"]]).max()
        target = float(args.pct_of_best) * float(max_final)

    b_conv = trials_to_threshold(b_curves, target)
    v_conv = trials_to_threshold(v_curves, target)
    b_conv["label"] = args.blabel
    v_conv["label"] = args.vlabel
    
    conv_table = pd.concat([b_conv, v_conv], ignore_index=True)
    conv_csv = os.path.join(args.out, "convergence.csv")
    conv_table.to_csv(conv_csv, index=False)
    
    # Convergence histogram
    plot_convergence_histogram(
        b_conv, v_conv, args.blabel, args.vlabel,
        os.path.join(args.out, "convergence_histogram.png")
    )

    # ========================================================================
    # 3) STABILITY ANALYSIS
    # ========================================================================
    print("[3/8] Computing stability metrics...")
    b_stab = stability_metrics(b_curves)
    v_stab = stability_metrics(v_curves)
    b_stab["label"] = args.blabel
    v_stab["label"] = args.vlabel
    
    stab_table = pd.concat([b_stab, v_stab], ignore_index=True)
    stab_csv = os.path.join(args.out, "stability.csv")
    stab_table.to_csv(stab_csv, index=False)

    # ========================================================================
    # 4) FINAL BEST DISTRIBUTION
    # ========================================================================
    print("[4/8] Computing final best distribution...")
    b_final = final_best_by_run(b_curves)
    v_final = final_best_by_run(v_curves)
    b_final["label"] = args.blabel
    v_final["label"] = args.vlabel
    
    plot_final_distribution(
        b_final, v_final, args.blabel, args.vlabel,
        os.path.join(args.out, "final_best_boxplot.png")
    )

    # ========================================================================
    # 5) STATISTICAL TESTS
    # ========================================================================
    print("[5/8] Running statistical tests...")
    
    stats_results = []
    
    # Test 1: Final best fitness
    stats_results.append(statistical_comparison(
        b_final["final_best"].values,
        v_final["final_best"].values,
        "final_best_fitness"
    ))
    
    # Test 2: Convergence speed
    stats_results.append(statistical_comparison(
        b_conv["trials_to_threshold"].values,
        v_conv["trials_to_threshold"].values,
        "trials_to_threshold"
    ))
    
    # Test 3: Stability (variance)
    stats_results.append(statistical_comparison(
        b_stab["variance"].values,
        v_stab["variance"].values,
        "stability_variance"
    ))
    
    # Test 4: Stability (CV)
    stats_results.append(statistical_comparison(
        b_stab["cv"].values,
        v_stab["cv"].values,
        "stability_cv"
    ))
    
    stats_df = pd.DataFrame(stats_results)
    stats_csv = os.path.join(args.out, "statistical_tests.csv")
    stats_df.to_csv(stats_csv, index=False)

    # ========================================================================
    # 6) DOMAIN-SPECIFIC METRICS (if params available)
    # ========================================================================
    print("[6/8] Extracting domain-specific metrics...")
    b_params = extract_domain_metrics(b)
    v_params = extract_domain_metrics(v)
    
    if b_params is not None and v_params is not None:
        # Get param keys
        param_keys = [k for k in b_params.columns 
                      if k not in ["trial", "run_id", "fitness"]]
        
        b_div = compute_diversity(b_params, param_keys)
        v_div = compute_diversity(v_params, param_keys)
        
        if b_div is not None and v_div is not None:
            plt.figure()
            plt.plot(b_div["trial"], b_div["param_diversity"], label=args.blabel)
            plt.plot(v_div["trial"], v_div["param_diversity"], label=args.vlabel)
            plt.xlabel("Trial")
            plt.ylabel("Parameter diversity (avg std)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "param_diversity.png"),
                        bbox_inches="tight")
            plt.close()


    # ========================================================================
    # 7) SUMMARY REPORT
    # ========================================================================
    print("[7/8] Generating summary report...")
    
    summary = {
        "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_label": args.blabel,
        "variant_label": args.vlabel,
        "convergence_threshold": target,
        
        # Final best
        "baseline_final_best_mean": float(b_final["final_best"].mean()),
        "baseline_final_best_std": float(b_final["final_best"].std()),
        "variant_final_best_mean": float(v_final["final_best"].mean()),
        "variant_final_best_std": float(v_final["final_best"].std()),
        
        # Convergence
        "baseline_mean_trials_to_threshold": float(np.nanmean(b_conv["trials_to_threshold"])),
        "variant_mean_trials_to_threshold": float(np.nanmean(v_conv["trials_to_threshold"])),
        
        # Stability
        "baseline_mean_variance": float(np.nanmean(b_stab["variance"])),
        "variant_mean_variance": float(np.nanmean(v_stab["variance"])),
        
        # Statistical significance
        "statistical_tests": stats_results,
    }
    
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ========================================================================
    # 8) LATEX TABLE (for thesis)
    # ========================================================================
    print("[8/8] Generating LaTeX table...")
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of " + args.blabel + " vs " + args.vlabel + "}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Metric & " + args.blabel + " & " + args.vlabel + " & Improvement & p-value \\\\",
        "\\hline",
    ]
    
    for result in stats_results:
        metric = result.get("metric","N/A").replace("_"," ").title()
        b_mean = result.get("baseline_mean", float("nan"))
        v_mean = result.get("variant_mean", float("nan"))
        pct = result.get("pct_improvement", float("nan"))
        pval = result.get("p_value_ttest", float("nan"))
        sig = "**" if result.get("significant_0.01", False) else ("*" if result.get("significant_0.05", False) else "")
        
        latex_lines.append(
            f"{metric} & {b_mean:.3f} & {v_mean:.3f} & {pct:+.1f}\\% & {pval:.4f}{sig} \\\\"
        )
    
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% * p < 0.05, ** p < 0.01"
    ])
    
    with open(os.path.join(args.out, "comparison_table.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {args.out}/")
    print("\nGenerated files:")
    print("  - evolution_with_ci.png       : Evolution curves with 95% CI")
    print("  - final_best_boxplot.png      : Distribution of final best")
    print("  - convergence_histogram.png   : Convergence speed distribution")
    print("  - param_diversity.png         : Exploration vs exploitation")
    print("  - statistical_tests.csv       : All statistical tests")
    print("  - convergence.csv             : Convergence metrics per run")
    print("  - stability.csv               : Stability metrics per run")
    print("  - summary.json                : Complete summary")
    print("  - comparison_table.tex        : LaTeX table for thesis")
    print("\nKey Results:")
    print(f"  Final Best: {b_final['final_best'].mean():.3f} → {v_final['final_best'].mean():.3f}")
    
    fb_test = [r for r in stats_results if r["metric"] == "final_best_fitness"][0]
    print(f"  Improvement: {fb_test['pct_improvement']:+.1f}% (p={fb_test['p_value_ttest']:.4f})")
    print(f"  Significant: {'YES' if fb_test['significant_0.05'] else 'NO'}")
    print("="*70)

if __name__ == "__main__":
    main()
