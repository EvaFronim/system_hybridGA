#!/usr/bin/env python3
"""Stage 5 fitness-ablation statistical analysis and plots.

Loads the neutral evaluation CSV produced by eval_correlation.py and runs
a full paired statistical analysis comparing GA outcomes across the four
fitness modes (F0–F3). Results are written to a text statistics file and
three publication-ready figures.

Execution flow:
    1. Load CSV and pivot to wide format (rows = runs, columns = F0–F3).
    2. Pairwise Wilcoxon signed-rank tests for incremental comparisons
       (F0→F1, F1→F2, F2→F3) with Holm-Bonferroni correction.
    3. Bootstrap confidence intervals for median differences (1000 iterations).
    4. Supplementary unpaired F0 vs F3 test (no correction; single comparison).
    5. Save statistics to STATS_FILE and figures to OUTPUT_DIR.

Statistical choices:
    - Wilcoxon signed-rank (non-parametric, paired) because n=15 per mode
      is too small to assume normality.
    - Median difference + bootstrap CI for consistency with Wilcoxon.
    - Cohen's d on paired differences (ddof=1) as a standardized effect size.
    - Holm-Bonferroni correction scoped to the 3 incremental comparisons only.
      The F0 vs F3 supplementary test is a separate question and is reported
      without correction.

Configuration:
    Exactly one METRIC_NAME/METRIC_LABEL/OUTPUT_DIR/STATS_FILE block in the
    Config section must be active (uncommented) at a time. All other blocks
    must remain commented out.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from typing import Dict, List

# ============================================================
# Configuration
# ============================================================

INPUT_CSV = 'logs/eval_correlation/stage5_eval_correlation.csv'


PLOT_DPI = 300
FONT_SIZE = 11
FIGURE_SIZE_SMALL = (8, 5)
FIGURE_SIZE_LARGE = (10, 6)

FITNESS_MODES = ['F0', 'F1', 'F2', 'F3']

# --- Active metric - uncomment exactly ONE block below. ---
# --- To switch metric: comment out the active block, uncomment the desired one. ---


#Primary: Forward progress
METRIC_NAME  = "assisted_forward_progress"
METRIC_LABEL = "Forward progress" #  (Σ max(forward_vel, 0)) higher is better
OUTPUT_DIR   = "results/stage5/_figures_forward_progress"
STATS_FILE = 'results/stage5/_statistics_forward.txt'

#METRIC_NAME  = "clean_forward_progress"
#METRIC_LABEL = "Clean forward progress"
#OUTPUT_DIR   = "results/stage5/_figures_clean_forward_progress"
#STATS_FILE   = "results/stage5/_statistics_clean_forward.txt"

#Secondary: Collisions (hit events)
#METRIC_NAME  = "assisted_crashes"
#METRIC_LABEL = "Collisions" # hit events lower is better
#OUTPUT_DIR   = "results/stage5/_figures_crashes"
#STATS_FILE = 'results/stage5/_statistics_Collisions.txt'

#METRIC_NAME  = "clean_crashes"
#METRIC_LABEL = "Clean collisions"
#OUTPUT_DIR   = "results/stage5/_figures_clean_crashes"
#STATS_FILE   = "results/stage5/_statistics_clean_crashes.txt"

#METRIC_NAME  = "clean_distance_pos"
#METRIC_LABEL = "Clean travelled distance"
#OUTPUT_DIR   = "results/stage5/_figures_clean_distance_pos"
#STATS_FILE   = "results/stage5/_statistics_clean_distance_pos.txt"

#Secondary: Lane keeping (mean |CTE|)
#METRIC_NAME  = "assisted_mean_abs_cte"
#METRIC_LABEL = "mean(|CTE|)" # Lane keeping: lower is better
#OUTPUT_DIR   = "results/stage5/_figures_mean_abs_cte"
#STATS_FILE = 'results/stage5/_statistics_lane_keeping.txt'

#METRIC_NAME  = "clean_mean_abs_cte"
#METRIC_LABEL = "Clean mean(|CTE|)"
#OUTPUT_DIR   = "results/stage5/_figures_clean_mean_abs_cte"
#STATS_FILE   = "results/stage5/_statistics_clean_lane_keeping.txt"

LOWER_IS_BETTER_METRICS = {
    "assisted_crashes",
    "assisted_mean_abs_cte",
    "clean_crashes",
    "clean_mean_abs_cte",
}

LOWER_IS_BETTER = METRIC_NAME in LOWER_IS_BETTER_METRICS


# Color palette per fitness mode (F0-F3)
MODE_COLORS = {
    "F0": "#1f77b4",  # blue
    "F1": "#ff7f0e",  # orange
    "F2": "#2ca02c",  # green
    "F3": "#d62728",  # red
}



def direction_from_diff(mean_diff: float) -> str:
    """Return better, worse, or same for the signed difference mean_diff = mode2 - mode1.

    Direction depends on the global LOWER_IS_BETTER flag:
        higher-is-better metrics: positive diff → 'better'
        lower-is-better metrics:  negative diff → 'better'
    """
    if abs(mean_diff) < 1e-12:
        return "same"
    if LOWER_IS_BETTER:
        return "better" if mean_diff < 0 else "worse"
    return "better" if mean_diff > 0 else "worse"


# ----- Holm-Bonferroni Correction

def holm_bonferroni(p_values, alpha=0.05):
    """
    Manual Holm-Bonferroni correction for family-wise error rate.
    
    Args:
        p_values: List of p-values to correct
        alpha: Family-wise error rate (default 0.05)
    
    Returns:
        reject: Boolean array indicating which nulls to reject
        p_corrected: Corrected p-values
    """
    n = len(p_values)
    indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[indices]
    
    reject = np.zeros(n, dtype=bool)
    p_corrected = np.ones(n)
    
    for i, p in enumerate(sorted_p):
        corrected = p * (n - i)
        p_corrected[indices[i]] = min(corrected, 1.0)
        
        if p < alpha / (n - i):
            reject[indices[i]] = True
        else:
            break  # Holm is sequential: once a hypothesis is not rejected,
                   # all remaining (larger) p-values are also not rejected.
    
    return reject, p_corrected


# ---- Data Loading

def load_and_pivot_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and pivot for paired analysis."""
    df = pd.read_csv(csv_path)
    
    print(f"[LOAD] Total rows: {len(df)}")
    print(f"[LOAD] Using metric: {METRIC_NAME}")
    
    # Normalize fitness_mode labels
    df['fitness_mode'] = df['fitness_mode'].astype(str).str.strip().str.upper()

    # Pivot to wide format
    pivoted = df.pivot(
        index='run_id',
        columns='fitness_mode',
        values=METRIC_NAME  
    )
    
    # Reindex to ensure all modes present
    pivoted = pivoted.reindex(columns=FITNESS_MODES)
    
    print(f"[PIVOT] Shape: {pivoted.shape}")
    print(f"[PIVOT] Complete runs: {pivoted.dropna().shape[0]}/{len(pivoted)}")
    
    return pivoted

# ---- Statistics

def paired_comparisons_with_correction(pivoted_df: pd.DataFrame) -> List[Dict]:
    """
    Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
    
    CORRECTION SCOPE: Applied to the 3 incremental comparisons (F0→F1, F1→F2, F2→F3)
    for the primary endpoint (METRIC_NAME)
    Note: results and p_values are built in sync only when all pairs have sufficient data. 
    If any pair is skipped (< 3 observations), the index mapping 
    between results and p_corrected may be incorrect.
    """
    results = []
    p_values = []
    
    for i in range(len(FITNESS_MODES) - 1):
        m1, m2 = FITNESS_MODES[i], FITNESS_MODES[i + 1]
        
        # Get paired data
        paired_data = pivoted_df[[m1, m2]].dropna()
        
        if len(paired_data) < 3:
            print(f"[WARN] Insufficient data for {m1} vs {m2}")
            continue
        
        g1 = paired_data[m1].values
        g2 = paired_data[m2].values
        
        # Wilcoxon signed-rank test (paired, two-sided)
        stat, p = wilcoxon(g1, g2, alternative='two-sided')
        p_values.append(p)
        
        # Descriptive statistics
        mean1, mean2 = g1.mean(), g2.mean()
        
        # Paired differences
        diffs = g2 - g1
        mean_diff = diffs.mean()
        median_diff = np.median(diffs)
        
        # Cohen's d for paired data (sample estimate, ddof=1)
        std_diff = diffs.std(ddof=1)
        
        # Guard against near-zero std (within-run identical values)
        if std_diff < 1e-12:
            cohens_d = 0.0
        else:
            cohens_d = mean_diff / std_diff
        
        # Percent change
        pct_change = (mean_diff / mean1 * 100) if abs(mean1) > 1e-12 else 0

        direction = direction_from_diff(mean_diff)
        
        results.append({
            'comparison': f'{m1}→{m2}',
            'n_paired': len(paired_data),
            'mean1': mean1,
            'mean2': mean2,
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'std_diff': std_diff,
            'pct_change': pct_change,
            'p_value': p,
            'cohens_d': cohens_d,
            'direction': direction,
        })
    
    # Holm-Bonferroni correction
    if len(p_values) > 0:
        reject, p_corrected = holm_bonferroni(p_values)
        
        for i, res in enumerate(results):
            res['p_corrected'] = p_corrected[i]
            res['reject_null'] = reject[i]
            
            # Significance markers
            p_cor = p_corrected[i]
            if p_cor < 0.001:
                res['significance'] = '***'
            elif p_cor < 0.01:
                res['significance'] = '**'
            elif p_cor < 0.05:
                res['significance'] = '*'
            else:
                res['significance'] = 'ns'
    
    return results


def bootstrap_paired_ci_median(pivoted_df: pd.DataFrame, 
                                mode1: str, mode2: str,
                                n_bootstrap: int = 1000,
                                alpha: float = 0.05) -> Dict:
    """
    Bootstrap confidence interval for MEDIAN difference (mode2 - mode1).
    Uses median to be consistent with Wilcoxon test philosophy.
    Bootstrap uses a fixed seed (42) for reproducibility, results are
    robust to seed choice given n_bootstrap=1000.
    
    Returns:
        dict with median_diff, ci_lower, ci_upper
    """
    paired_data = pivoted_df[[mode1, mode2]].dropna()
    
    if len(paired_data) < 3:
        return {'median_diff': 0, 'ci_lower': 0, 'ci_upper': 0}
    
    g1 = paired_data[mode1].values
    g2 = paired_data[mode2].values
    
    median_diffs_bootstrap = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Resample runs with replacement
        indices = np.random.choice(len(paired_data), size=len(paired_data), replace=True)
        g1_boot = g1[indices]
        g2_boot = g2[indices]
        
        # MEDIAN difference (not mean!)
        median_diffs_bootstrap.append(np.median(g2_boot - g1_boot))
    
    median_diffs_bootstrap = np.array(median_diffs_bootstrap)
    
    return {
        'median_diff': np.median(median_diffs_bootstrap),
        'ci_lower': np.percentile(median_diffs_bootstrap, 100 * alpha / 2),
        'ci_upper': np.percentile(median_diffs_bootstrap, 100 * (1 - alpha / 2)),
    }


def overall_f0_vs_f3_test(pivoted_df: pd.DataFrame) -> Dict:
    """
    Supplementary overall test: F0 vs F3 (paired Wilcoxon).
    """
    paired_data = pivoted_df[['F0', 'F3']].dropna()
    
    if len(paired_data) < 3:
        print("[WARN] Insufficient data for F0 vs F3 overall test")
        return {}
    
    g0 = paired_data['F0'].values
    g3 = paired_data['F3'].values
    
    # Paired Wilcoxon
    stat, p = wilcoxon(g0, g3, alternative='two-sided')
    
    # Descriptive
    diffs = g3 - g0
    mean_diff = diffs.mean()
    median_diff = np.median(diffs)
    std_diff = diffs.std(ddof=1)
    direction = direction_from_diff(mean_diff)
    cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0
    pct_change = (mean_diff / g0.mean() * 100) if abs(g0.mean()) > 1e-12 else 0
    
    # Bootstrap CI for median difference
    boot_ci = bootstrap_paired_ci_median(pivoted_df, 'F0', 'F3')
    
    # Significance (no correction, single test)
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    return {
        'comparison': 'F0→F3 (overall)',
        'n_paired': len(paired_data),
        'mean1': g0.mean(),
        'mean2': g3.mean(),
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'std_diff': std_diff,
        'pct_change': pct_change,
        'p_value': p,
        'cohens_d': cohens_d,
        'boot_median_diff': boot_ci['median_diff'],
        'boot_ci_lower': boot_ci['ci_lower'],
        'boot_ci_upper': boot_ci['ci_upper'],
        "direction": direction,
        'significance': sig,
    }


def plot_boxplot(pivoted_df, output_dir):
    """Colored boxplot (one color per mode)."""
    plt.figure(figsize=FIGURE_SIZE_SMALL)

    bp = plt.boxplot(
        [pivoted_df[mode].dropna() for mode in FITNESS_MODES],
        tick_labels=FITNESS_MODES,
        patch_artist=True,
        widths=0.6
    )

    # Color each box by mode
    for patch, mode in zip(bp['boxes'], FITNESS_MODES):
        patch.set_facecolor(MODE_COLORS.get(mode, "#cccccc"))
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Keep lines readable
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)

    plt.xlabel('Fitness Complexity', fontsize=FONT_SIZE+1, fontweight='bold')
    plt.ylabel(METRIC_LABEL, fontsize=FONT_SIZE+1, fontweight='bold')
    plt.title('GA Outcomes Under Different Fitness Objectives',
              fontsize=FONT_SIZE+2, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] boxplot.png")


def plot_trajectory(pivoted_df, output_dir):
    """Mean ± SEM trajectory across fitness modes (colored markers per mode, black line = mean)."""
    plt.figure(figsize=FIGURE_SIZE_SMALL)

    means = pivoted_df[FITNESS_MODES].mean()
    stds = pivoted_df[FITNESS_MODES].std()
    counts = pivoted_df[FITNESS_MODES].count()
    sems = stds / np.sqrt(counts)

    x = np.arange(len(FITNESS_MODES))

    # Error bars (neutral color)
    plt.errorbar(x, means, yerr=sems, fmt='-',
                 linewidth=2, capsize=5, capthick=1.5,
                 color='black', label='Mean ± SEM', zorder=5)

    # Colored markers per mode
    for i, mode in enumerate(FITNESS_MODES):
        plt.plot(i, means[mode], marker='o', markersize=10,
                 color=MODE_COLORS.get(mode, "black"), zorder=10)

    plt.xticks(x, FITNESS_MODES, fontsize=FONT_SIZE)
    plt.xlabel('Fitness Complexity', fontsize=FONT_SIZE+1, fontweight='bold')
    plt.ylabel(f'{METRIC_LABEL} (Mean ± SEM)', fontsize=FONT_SIZE+1, fontweight='bold')
    plt.title('Performance Trajectory Across Fitness Modes',
              fontsize=FONT_SIZE+2, fontweight='bold')
    plt.legend(loc='best', fontsize=FONT_SIZE-1)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] trajectory.png")


def plot_paired_lines(pivoted_df, output_dir):
    """Spaghetti plot: gray runs, colored mean markers."""
    plt.figure(figsize=FIGURE_SIZE_LARGE)

    # Individual runs (light gray)
    for run_id in pivoted_df.index:
        values = pivoted_df.loc[run_id, FITNESS_MODES]
        if not values.isna().all():
            plt.plot(range(len(FITNESS_MODES)), values, 'o-',
                    alpha=0.25, color='#808080', linewidth=1, markersize=4)

    means = pivoted_df[FITNESS_MODES].mean()

    # Mean line (black)
    plt.plot(range(len(FITNESS_MODES)), means, '-',
            linewidth=3, color='black', label='Mean', zorder=10)

    # Mean markers (colored per mode)
    for i, mode in enumerate(FITNESS_MODES):
        plt.plot(i, means[mode], marker='o', markersize=12,
                 color=MODE_COLORS.get(mode, "black"), zorder=20)

    plt.xticks(range(len(FITNESS_MODES)), FITNESS_MODES, fontsize=FONT_SIZE)
    plt.xlabel('Fitness Complexity', fontsize=FONT_SIZE+1, fontweight='bold')
    plt.ylabel(METRIC_LABEL, fontsize=FONT_SIZE+1, fontweight='bold')
    plt.title('Individual Runs Across Fitness Modes (Paired Data)',
              fontsize=FONT_SIZE+2, fontweight='bold')
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paired_lines.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] paired_lines.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("STAGE 5: PUBLICATION-READY PAIRED ANALYSIS")
    print("=" * 70)
    
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['font.family'] = 'sans-serif'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    pivoted_df = load_and_pivot_data(INPUT_CSV)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    summary = pivoted_df[FITNESS_MODES].describe().T
    summary['cv'] = summary['std'] / summary['mean']
    print(summary)
    
    # Paired comparisons
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS (Wilcoxon Signed-Rank + Holm-Bonferroni)")
    print("=" * 70)
    
    paired_results = paired_comparisons_with_correction(pivoted_df)
    
    # Bootstrap CIs (for MEDIAN differences)
    print("\nComputing bootstrap CIs for median differences...")
    for res in paired_results:
        comp = res['comparison']
        m1, m2 = comp.split('→')
        boot_ci = bootstrap_paired_ci_median(pivoted_df, m1, m2)
        res['boot_median_diff'] = boot_ci['median_diff']
        res['boot_ci_lower'] = boot_ci['ci_lower']
        res['boot_ci_upper'] = boot_ci['ci_upper']
    
    # Supplementary overall test: F0 vs F3 (not part of Holm correction)
    print("\nComputing supplementary F0 vs F3 overall test...")
    overall_result = overall_f0_vs_f3_test(pivoted_df)
    
    # Write statistics file
    with open(STATS_FILE, 'w') as f:
        def tee(msg):
            print(msg)
            f.write(msg + '\n')
        
        tee("\n" + "=" * 70)
        tee(f"SUMMARY STATISTICS ({METRIC_NAME})")
        tee("=" * 70)
        tee(summary.to_string())
        
        tee("\n" + "=" * 70)
        tee("PAIRWISE COMPARISONS (Wilcoxon + Holm-Bonferroni)")
        tee("=" * 70)
        tee(f"{'Comp':<8} {'n':>4} {'Mean1':>7} {'Mean2':>7} {'MnΔ':>7} {'MdΔ':>7} "
            f"{'Δ%':>7} {'p_raw':>9} {'p_adj':>9} {'d':>7} {'95% CI (MdΔ)':>22} "
            f"{'Dir':>8} {'Sig':>4}")
        tee("-" * 110)
        
        for r in paired_results:
            ci_str = f"[{r['boot_ci_lower']:>7.2f}, {r['boot_ci_upper']:>7.2f}]"
            tee(f"{r['comparison']:<8} {r['n_paired']:>4} "
                f"{r['mean1']:>7.2f} {r['mean2']:>7.2f} "
                f"{r['mean_diff']:>+7.2f} {r['median_diff']:>+7.2f} "
                f"{r['pct_change']:>+6.1f}% {r['p_value']:>9.6f} "
                f"{r['p_corrected']:>9.6f} {r['cohens_d']:>7.3f} "
                f"{ci_str:>22} {r['direction']:>8} {r['significance']:>4}")
        
        # Supplementary overall F0 vs F3 test (separate from incremental Holm scope)
        if overall_result:
            tee("\n" + "=" * 70)
            tee("SUPPLEMENTARY OVERALL TEST: F0 vs F3 (Wilcoxon, no correction)")
            tee("=" * 70)
            tee("Answers a different question than the incremental tests:")
            tee("'Is the full F3 fitness overall better than F0 baseline?'")
            tee("")
            tee(f"{'Comp':<18} {'n':>4} {'Mean1':>7} {'Mean2':>7} {'MnΔ':>7} {'MdΔ':>7} "
                f"{'Δ%':>7} {'p':>9} {'d':>7} {'95% CI (MdΔ)':>22} {'Sig':>4}")
            tee("-" * 110)
            r = overall_result
            ci_str = f"[{r['boot_ci_lower']:>7.2f}, {r['boot_ci_upper']:>7.2f}]"
            tee(f"{r['comparison']:<18} {r['n_paired']:>4} "
                f"{r['mean1']:>7.2f} {r['mean2']:>7.2f} "
                f"{r['mean_diff']:>+7.2f} {r['median_diff']:>+7.2f} "
                f"{r['pct_change']:>+6.1f}% {r['p_value']:>9.6f} "
                f"{r['cohens_d']:>7.3f} "
                f"{ci_str:>22} {r['direction']:>8} {r['significance']:>4}")
        
        tee("\n" + "=" * 70)
        tee("NOTES")
        tee("=" * 70)
        
        # Dynamic description per metric
        metric_descriptions = {
            "assisted_forward_progress": "Sum of max(forward_vel, 0) over the episode (higher is better), averaged across repeats.",
            "assisted_crashes": "Number of hit events per episode (from info['hit'] transitions), averaged across repeats.",
            "assisted_mean_abs_cte": "Mean absolute cross-track error per episode (lower is better), averaged across repeats.",
            "assisted_distance": "Travelled distance from (x,z) position deltas (higher is better), averaged across repeats.",
            "assisted_lap": "Final lap count per episode (higher is better), averaged across repeats.",
            "assisted_mean_speed": "Mean speed over the episode, averaged across repeats.",
            "clean_forward_progress": "Sum of max(forward_vel, 0) under clean_policy_rollout, without assisted steering/throttle corrections (higher is better), averaged across repeats.",
            "clean_distance_pos": "Travelled distance from (x,z) position deltas under clean_policy_rollout (higher is better), averaged across repeats.",
            "clean_crashes": "Number of hit events under clean_policy_rollout (lower is better), averaged across repeats.",
            "clean_mean_speed": "Mean speed under clean_policy_rollout, averaged across repeats.",
            "clean_mean_abs_cte": "Mean absolute cross-track error under clean_policy_rollout (lower is better), averaged across repeats.",
            "clean_steps_survived": "Number of simulator steps survived under clean_policy_rollout (higher is better), averaged across repeats.",
        }
        desc = metric_descriptions.get(
            METRIC_NAME,
            "Custom telemetry-based metric, averaged across repeats."
        )
        tee(f"Metric: {METRIC_NAME} (Neutral evaluation from telemetry)")
        tee(f"Computed as: {desc}")

        tee(f"MnΔ = Mean difference, MdΔ = Median difference")
        tee(f"d = Cohen's d (paired, ddof=1)")
        tee(f"95% CI computed via bootstrap (1000 iterations) for median difference")
        tee(f"Holm-Bonferroni correction applied across 3 incremental comparisons")
        tee(f"Supplementary F0 vs F3 test reported separately (no correction)")
        tee(f"Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
        tee("\n" + "=" * 70)
    
    print(f"\n[SAVE] {STATS_FILE}")
    
    # Generate plots
    plot_boxplot(pivoted_df, OUTPUT_DIR)
    plot_trajectory(pivoted_df, OUTPUT_DIR)
    plot_paired_lines(pivoted_df, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("COMPLETED!")
    print(f"Outputs: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()