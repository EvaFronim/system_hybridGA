# evolve_ga.py  GA-only for DonkeyCar
# ==========================================================================
# - Unified HyperParams
# - Preprocess/Detectors/Governor fully parameterized
# - Fitness: stage weights/targets/penalties in config
# - RL write-back with validation (multi-eval) + rollback
# - Elite protection: protected top-K + Hall-of-Fame by weight hash
# - Fitness consistency: re-eval top-fraction, trimmed mean aggregation
# - Gentler, EMA-smoothed adaptive hyperparams (lr/mutation) with clamps
# - Population Health monitor + conservative curriculum + gated interventions
# - Intervention cooldown; antithetic seeds option for eval stability
# ==========================================================================

from __future__ import annotations

import os
import random
import time
import json
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Mapping, Callable
import numpy as np
import copy
import cv2 

try:
    import gym
except Exception:  # gymnasium fallback
    import gymnasium as gym  # type: ignore

try:
    import gym_donkeycar  # noqa: F401
except Exception:
    pass

import torch

from .genome import Genome
from .controller import Controller  

VERBOSE = True             # True = more debug
SHOW_DEBUG_WINDOWS = False  # True = heavy cv2 windows

def log(msg: str):
    if VERBOSE:
        print(msg)

PRINT_GENOME_SCORES = True  # Set False to disable per-genome score logs.

def _log_genome_score(gen_idx: int, gen_num: int, stage_name: str, fitness: float, metrics: dict | None, pop_size: int):
    """
    Prints a compact text summary of the scores and metrics for a single genome.

    It extracts individual performance factors like distance and speed, and formats 
    them into a clean console log line if logging is enabled.
    """
    try:
        m = metrics or {}
        
        # Extract individual metrics using safe fallbacks and explicit type casting
        d  = float(m.get("distance",    m.get("dist", 0.0)) or 0.0)
        sv = float(m.get("survival",    0.0) or 0.0)
        sp = float(m.get("speed",       0.0) or 0.0)
        th = float(m.get("throttle",    0.0) or 0.0)
        sm = float(m.get("smoothness",  0.0) or 0.0)
        ln = float(m.get("lane",        0.0) or 0.0)
        tn = float(m.get("turns",       0.0) or 0.0)
        steps = int(m.get("steps",      m.get("n_steps", 0)) or 0)

        # Print the formatted summary line only if the global display flag is active
        if PRINT_GENOME_SCORES:
            log(f"[Gen {gen_num:02d} | {stage_name}] "
                f"{gen_idx+1:02d}/{pop_size:02d} fit={fitness:.2f} "
                f"(dist={d:.2f}, surv={sv:.2f}, spd={sp:.2f}, thr={th:.2f}, "
                f"smooth={sm:.2f}, lane={ln:.2f}, turns={tn:.2f}, steps={steps})")
                
    except Exception as e:
        log(f"i per-genome logging skipped: {e}")


def seed_everything(seed: int = 42) -> None:
    """
    Seeds all random number generators to ensure training runs are reproducible.

    It locks the randomness for Python, NumPy, and PyTorch on both CPU and GPU, 
    and disables non-deterministic GPU optimization algorithms.

    Args:
        seed: The integer value used to initialize the random states.
    """
    # Set environment variable for Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Seed core libraries for CPU-bound processes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Seed CUDA calculations if a graphics card is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        # Enforce deterministic algorithms inside PyTorch backend operators
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Disable lower-precision TF32 math to keep exact arithmetic stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        # Pass silently if the active PyTorch version does not support these flags
        pass


@dataclass
class HyperParams:

    calib_spread_thresh: float = 45.0
    calib_frames: int = 30
    calib_edge_density_lo: float = 0.02
    calib_edge_density_hi: float = 0.25

    imitation_lr:float = 5e-4
    imitation_min_conf: float = 0.20
    imitation_max_samples: int = 256
    imitation_epochs: int = 1

    use_sym_control: bool = False
    dash_score_thr: float = 0.45
    sym_conf_thr: float = 0.50
    dash_kp: float = 0.45
    sym_kp_off: float = 0.30
    sym_kp_head: float = 0.10
    # Preprocess
    crop_top_ratio: float = 0.40 
    resize_w: int = 80
    resize_h: int = 60
    blur_ksize: int = 3 # Gaussian blur kernel size
    canny_low: int = 20
    canny_high: int = 85

    # Dynamically adjusts top crop based on edge density and diagonal lane cues.
    adaptive_crop_enabled: bool = False
    ac_min_ratio: float = 0.30    
    ac_max_ratio: float = 0.50    
    ac_default_ratio: float = 0.35   
    ac_edge_band_lo: float = 0.40  
    ac_edge_band_hi: float = 0.95  
    ac_edge_density_thr: float = 0.10  
    ac_diag_need_lines: int = 1   
    ac_loosen_when_blind: float = -0.05  
    ac_tighten_when_clear: float = +0.05 

    # Image filtering
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    auto_canny: bool = True
    auto_canny_p1: int = 25
    auto_canny_p2: int = 75
    morph_in_preprocess: bool = True   # Apply morphology during CNN preprocessing.
    use_morph_close: bool = True       
    morph_kernel: int = 3               
    morph_iters: int = 1                

    # Lane/turn detectors
    side_line_row_start: float = 0.70
    diag_roi_row_start: float = 0.49 
    diag_roi_col_lo: float = 0.15   
    diag_roi_col_hi: float = 0.95   

    flat_roi_row_start: float = 0.70 
    flat_roi_col_lo: float = 1/3 
    flat_roi_col_hi: float = 2/3 
    hough_threshold: int = 14 
    diag_min_len: int = 8 
    diag_max_gap: int = 10 
    flat_min_len: int = 19
    flat_max_gap: int = 4 
    straight_angle_deg: float = 25.0 
    diag_angle_min: float = 60.0  
    diag_angle_max: float = 85.0  
    edge_bin_thresh: float = 0.20 

    # Turn confidence and smoothing
    turninfo_roi_start: float = 0.50 
    turninfo_imbalance_gain: float = 1.7 
    turninfo_total_edges_min: int = 6 
    lane_conf_norm: float = 0.04  
    min_diag_votes: int = 1
    turn_smooth_alpha: float = 0.30
    edge_dir_bias: float = 1.13 
    overall_conf_turn_cut: float = 0.35
    straight_boost_conf: float = 0.68

    # Side-line detection
    side_line_pixel_threshold: int = 130
    side_line_min_count: int = 31

    # Off-lane penalty
    offlane_thr_cut: float = 0.05 
    offlane_penalty_step: float = 0.05

    # Lane curvature control
    lane_conf_thr: float = 0.50
    lane_kp_off:  float = 0.35
    lane_kp_head: float = 0.12
    lane_kp_curv: float = 0.10
    offset_sign:  float = 1.0    # Use -1.0 if lateral offset sign is inverted.
    curv_thr_cap: float = 1.0
    curv_thr_slow: float = 0.80

    # Dynamic lane-width scaling 
    use_dynamic_lane_width: bool = True
    ref_lane_w_px: Optional[float] = None  # None -> defaults to 0.60 * resize_w

    # Turn bonus (positive reward when turning correctly)
    turn_bonus_min_steer: float = 0.15      
    turn_bonus_intensity_ref: float = 0.50  
    turn_bonus_conf_ref: float = 0.60       
    turn_bonus_norm: float = 0.40          
    turn_bonus_mix: float = 0.30            

    step_micro_reward: float = 0.02 

    # Governor
    steer_bracket_hi: float = 0.7 
    steer_bracket_mid: float = 0.4 
    scale_hi: float = 0.80 
    scale_mid: float = 0.80 
    turn_intensity_cut: float = 0.40 
    smooth_up_expert: float = 0.12 
    smooth_up_default: float = 0.08 
    smooth_down: float = 0.25 
    fast_path_turn_mult: float = 0.90

    # --- Elite RL knobs ---
    elite_rl_enabled: bool = False     
    elite_rl_frac: float = 0.20         
    elite_eval_runs: int = 2           
    elite_rl_steps: int = 1000            
    elite_accept_margin: float = 0.01

    stage_min_throttle: dict[str, float] = field(default_factory=lambda: {
        "Basic": 0.06,
        "Early-Intermediate": 0.08,
        "Late-Intermediate": 0.10,
        "Advanced": 0.12,
        "Expert": 0.14,
    })

    # Fitness scaling targets
    dist_scaling_by_stage: dict[str, float] = field(default_factory=lambda: {
        "Basic": 22.0,
        "Early-Intermediate": 30.0,
        "Late-Intermediate": 40.0,
        "Advanced": 55.0,
        "Expert": 70.0,
    })

    # Mean throttle targets
    speed_target_by_stage: dict[str, float] = field(default_factory=lambda: {
        "Basic": 0.08,
        "Early-Intermediate": 0.10,
        "Late-Intermediate": 0.12,
        "Advanced": 0.14,
        "Expert": 0.20,
    })

    smoothness_norm: float = 0.25

    # Debug overlay flags
    debug_overlay: bool = False
    debug_overlay_every_n: int = 5         
    debug_overlay_output: str = "window"   
    debug_overlay_scale: float = 1.0
    sroi_world_coords: bool = True        

    # Penalties / thresholds
    idle_reward_cut: float = 0.01 
    slow_thr_cut: float = 0.10
    slow_steer_cut: float = 0.2
    slow_steps_grace: int = 5 
    slow_penalty_step: float = 0.02 
    slow_penalty_cap_steps: int = 10 
    curve_turn_penalty_gain: float = 1.0 
    harsh_turn_thr: float = 0.5
    harsh_thr_thr: float = 0.4

    # Sampling / action stride
    action_stride_ga: int = 5
    action_stride_rl: int = 5 

    # log_std clamps (controller prior)
    steer_logstd_min: float = -2.3
    steer_logstd_max: float = -0.9
    thr_logstd_min: float = -2.3
    thr_logstd_max: float = -0.7
    steer_ema_keep: float = 0.5

    # Hough-based steering helper
    use_hough_helper: bool = True
    hough_kp_dir: float = 0.40
    min_turn_steer: float = 0.25

    # Sign conventions
    steer_sign: int = 1
    dir_sign_gain: float = 1.0

    # Input augmentation
    aug_prob: float = 0.0
    canny_jitter_pct: float = 0.15
    brightness_jitter_pct: float = 0.12
    contrast_jitter_pct: float = 0.12
    blur_k_choices: tuple[int, ...] = (0, 3)

    # Adaptive Hough threshold
    hough_min_threshold: int = 12
    hough_adapt_step: int = 3

@dataclass
class PopulationHealth:
    """Summary metrics describing population diversity and stability."""
    genetic_diversity: float = 0.0
    behavioral_diversity: float = 0.0
    fitness_cv: float = 0.0
    collapse_risk: float = 0.0
    score: float = 1.0  # higher is healthier

@dataclass
class EliteProtection:
    """Controls protection of top individuals from RL fine-tuning."""
    top_k: int = 1 
    min_generations_before_rl: int = 1 
    hof_max: int = 5

@dataclass
class AdaptiveConfig:
    """
    Holds all configuration parameters for adaptive genetic algorithm training.
    
    It manages population scaling, mutation rules, training stages, and 
    automated intervention gates to prevent population collapse.
    """

    # Population parameters
    min_population: int = 10
    max_population: int = 50
    population_size: int = 15

    # Exploration burst parameters used when stagnation occurs
    explore_burst_period: int = 5
    explore_burst_mrate_mult: float = 1.20
    explore_burst_entropy_mult: float = 1.25
    try_accept_worse_prob: float = 0.15

    elite_ratio: float = 0.35

    # Baseline mutation behaviors
    base_mutation_rate: float = 0.10
    base_mutation_strength: float = 0.15
    mutate_head_k: int = 1
    mutate_gaussian: bool = True

    # Curriculum activation state
    curriculum_enabled: bool = True
    curriculum_stages: Optional[List[Dict[str, Any]]] = None

    # Early stopping limits
    patience: int = 8
    min_improvement: float = 0.009

    # Evaluation seeds
    seed: int = 42
    use_antithetic_eval_seeds: bool = True

    # Dynamic fitness function component weights per training stage
    stage_weights: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "Basic": {
            "distance": 0.40, "survival": 0.36, "speed": 0.08,
            "smoothness": 0.08, "lane": 0.08, "turns": 0.0,
        },
        "Early-Intermediate": {
            "distance": 0.37, "survival": 0.35, "speed": 0.10,
            "smoothness": 0.08, "lane": 0.10, "turns": 0.0,
        },
        "Late-Intermediate": {
            "distance": 0.32, "survival": 0.32, "speed": 0.12,
            "smoothness": 0.12, "lane": 0.12, "turns": 0.0,
        },
        "Advanced": {
            "distance": 0.30, "survival": 0.26, "speed": 0.18,
            "smoothness": 0.14, "lane": 0.12, "turns": 0.0,
        },
        "Expert": {
            "distance": 0.30, "survival": 0.20, "speed": 0.20,
            "smoothness": 0.16, "lane": 0.14, "turns": 0.0,
        },
    })

    # Sub-config classes
    elite_protection: EliteProtection = field(default_factory=EliteProtection)

    # Fitness stability variables across multiple rollouts
    fitness_rollouts_top_frac: float = 0.40
    fitness_rollouts_top_n: int = 2
    fitness_trim_ratio: float = 0.10

    # Learning rate decay and exponential moving average parameters
    mrate_decay_per_gen: float = 0.995
    ema_alpha: float = 0.20
    mrate_min: float = 0.05

    # Population health metrics thresholds for triggers
    intervention_enabled: bool = True
    intervention_cooldown_gens: int = 5
    collapse_risk_gate: float = 0.60
    health_score_gate: float = 0.30

    # Hyperparameter subgroup reference
    hparams: HyperParams = field(default_factory=HyperParams)

    # Internal tracking value for smoothed mutation rate
    _ema_mut_rate: Optional[float] = None

    def __post_init__(self) -> None:
        """
        Initializes default curriculum stages and saves the starting mutation rate.
        """
        if self.curriculum_stages is None:
            # Set up the default progression stages 
            self.curriculum_stages = [
                {"max_steps": 1000, "max_throttle": 0.10, "adaptive_throttle": True, "name": "Basic"},
                {"max_steps": 1000, "max_throttle": 0.15, "adaptive_throttle": True, "name": "Early-Intermediate"},
                {"max_steps": 1000, "max_throttle": 0.20, "adaptive_throttle": True, "name": "Late-Intermediate"},
                {"max_steps": 1000, "max_throttle": 0.25, "adaptive_throttle": True, "name": "Advanced"},
                {"max_steps": 1000, "max_throttle": 0.50, "adaptive_throttle": True, "name": "Expert"},
            ]

        # Sync the initial tracking mutation rate with the configured base rate
        self._ema_mut_rate = self.base_mutation_rate

# ---------- DIVERSITY / HEALTH 

class DiversityEvaluator:
    """
    Calculates different types of diversity measures for a population of genomes.
    """
    @staticmethod
    def diversity_calculation(population: List[Genome],
                              actions: Optional[List[List[Tuple[float, float]]]] = None
                              ) -> Dict[str, float]:
        """
        Computes genetic, fitness, and behavioral diversity metrics for the population.

        Args:
            population: A list of genomes to analyze.
            actions: Optional list of driving actions for each genome.

        Returns:
            A dictionary containing genetic, behavioral, and fitness scores.
        """
        if len(population) < 2:
            return {"genetic": 0.0, "behavioral": 0.0, "fitness": 0.0}

        # 1) Genetic diversity: sample up to 1000 weights to save time
        genetic_diversity = 0.0
        first_w = getattr(population[0], "cnn_weights", None)
        if first_w is not None and len(first_w) > 100:
            W = len(first_w)
            if all(getattr(g, "cnn_weights", None) is not None and len(g.cnn_weights) == W for g in population):
                S = min(1000, W)
                rng = np.random.RandomState(42)
                idx = rng.choice(W, size=S, replace=False)
                sample_weights = np.array(
                    [[population[p].cnn_weights[i] for p in range(len(population))] for i in idx],
                    dtype=np.float64
                )
                # Replace invalid values with NaN and calculate statistical variance
                sw = np.where(np.isfinite(sample_weights), sample_weights, np.nan)
                row_vars = np.nanvar(sw, axis=1)
                row_vars = row_vars[np.isfinite(row_vars)]
                genetic_diversity = float(np.mean(row_vars)) if row_vars.size else 0.0

        # 2) Fitness diversity: calculate the coefficient of variation
        fit = np.array([float(getattr(g, "fitness", 0.0) or 0.0) for g in population], dtype=np.float32)
        _fit = [float(f) for f in fit if np.isfinite(f)]
        fitness_diversity = float(np.std(_fit) / (np.mean(_fit) + 1e-8)) if _fit else 0.0

        # 3) Behavioral diversity: compare differences in steering histograms using Jensen-Shannon
        behavioral_diversity = 0.0
        if actions:
            hists: List[np.ndarray] = []
            for acts in actions:
                if not acts:
                    continue
                steer_vals = [a[0] for a in acts]
                if not steer_vals:
                    continue
                # Create a 20-bin histogram for steering inputs between -1.0 and 1.0
                h, _ = np.histogram(steer_vals, bins=20, range=(-1.0, 1.0))
                p = (h.astype(np.float64) + 1e-12)
                p /= p.sum()
                hists.append(p)
                
            if len(hists) >= 2:
                def _kl(p, q):
                    # Kullback-Leibler divergence helper function
                    return float(np.sum(p * (np.log(p) - np.log(q + 1e-12))))

                def _js(p, q):
                    # Jensen-Shannon divergence helper function
                    m = 0.5 * (p + q)
                    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

                # Compare every genome pair to compute average behavioral distance
                total, cnt = 0.0, 0
                for i in range(len(hists)):
                    for j in range(i + 1, len(hists)):
                        total += _js(hists[i], hists[j])
                        cnt += 1
                if cnt > 0:
                    behavioral_diversity = float(total / cnt)

        return {"genetic": genetic_diversity, "behavioral": behavioral_diversity, "fitness": fitness_diversity}


def evaluate_population_health(population: List[Genome],
                               actions: Optional[List[List[Tuple[float, float]]]] = None) -> PopulationHealth:
    """
    Calculates the diversity and collapse risk of the current population.

    It analyzes genetic structure, driving behavior, and fitness variance 
    to output a general health score and determine if the population is stagnating.

    Args:
        population: A list of genomes to evaluate.
        actions: Optional list of driving actions taken by the genomes.

    Returns:
        A PopulationHealth object containing metrics, collapse risk, and total score.
    """
    # Fetch raw diversity metrics from the analyzer class
    div = DiversityEvaluator.diversity_calculation(population, actions=actions)
    g = div["genetic"]
    b = div["behavioral"]
    fcv = div["fitness"]
    
    # Normalize inputs into a 0.0 to 1.0 range using logistic S-curves
    g_n = 1.0 / (1.0 + math.exp(-1e3 * (g - 5e-4)))
    b_n = 1.0 / (1.0 + math.exp(-10.0 * (b - 0.02)))
    fcv_n = 1.0 / (1.0 + math.exp(-5.0 * (fcv - 0.15)))
    
    # Calculate global collapse risk using a weighted formula
    collapse = 1.0 - (0.4 * g_n + 0.4 * b_n + 0.2 * fcv_n)
    health_score = max(0.0, 1.0 - collapse)
    
    return PopulationHealth(
        genetic_diversity=g, 
        behavioral_diversity=b, 
        fitness_cv=fcv,
        collapse_risk=float(max(0.0, min(1.0, collapse))), 
        score=float(health_score)
    )

# ---------- CURRICULUM 

class CurriculumManager:
    """
    Manages the training stages (Curriculum Learning) for a population.
    
    It checks if the population drives well enough to move to the next, 
    more difficult track stage, or advances them automatically if they get stuck.
    """
    def __init__(self, config: AdaptiveConfig):
        """
        Initializes the curriculum manager with a configuration object.

        Args:
            config (AdaptiveConfig): The configuration containing the stage definitions.
        """
        self.config = config
        self.current_stage = 0
        self.stage_generations = 0
        self.stage_patience = 6
        self.performance_history: List[Dict[str, float]] = []

    def get_current_stage(self) -> Dict[str, Any]:
        """
        Returns the dictionary settings of the currently active training stage.

        Returns:
            Dict[str, Any]: Configuration parameters for the current stage.
        """
        return self.config.curriculum_stages[self.current_stage]

    def should_advance(self, population_fitness: List[float]) -> bool:
        """
        Analyzes the latest fitness scores to decide if it is time to move 
        to the next training stage.

        To advance, the population must beat the stage thresholds for 4 
        consecutive generations. If they take too long, a timeout triggers an automatic advance.

        Args:
            population_fitness (List[float]): A list of fitness scores from the current generation.

        Returns:
            bool: True if the population is ready to advance (or timed out), False otherwise.
        """
        # Guard clause: Bypass evaluation if curriculum is disabled or final stage is reached
        if (not self.config.curriculum_enabled) or (self.current_stage >= len(self.config.curriculum_stages) - 1):
            return False

        # Filter out invalid or infinite numerical evaluations from raw scoring arrays
        _pf = [float(f) for f in (population_fitness or []) if np.isfinite(f)]
        mean_f = float(np.mean(_pf)) if _pf else 0.0
        top_f   = float(np.max(_pf)) if _pf else float('-inf')
        top3 = float(np.mean(sorted(_pf, reverse=True)[:3])) if _pf else 0.0
        bottom = float(np.min(_pf)) if _pf else float('inf')
        
        # Append latest performance record and maintain a rolling window max size of 10
        self.performance_history.append({"mean": mean_f, "top": top_f, "top3": top3, "bottom": bottom})
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
            
        # Target threshold grids optimized per stage tier (compensating for rising environment difficulty)
        # NOTE: keys are positional indices into config.curriculum_stages, not stage names.
        # If you add/remove/reorder stages in AdaptiveConfig.__post_init__, update this mapping too.
        thresholds = {
            0: {"mean": 72.0, "top": 85.0, "top3": 85.0, "bottom": 0.0},
            1: {"mean": 70.0, "top": 80.0, "top3": 80.0, "bottom": 0.0},
            2: {"mean": 68.0, "top": 75.0, "top3": 75.0, "bottom": 0.0},
            3: {"mean": 66.0, "top": 70.0, "top3": 70.0, "bottom": 0.0},
            4: {"mean": 64.0, "top": 70.0, "top3": 68.0, "bottom": 0.0},
        }
        th = thresholds.get(self.current_stage, {"mean": 1e9, "top": 1e9, "top3": 1e9, "bottom": 1e9})

        # Conservative Rule: Require continuous target compliance across the last 4 generations
        if len(self.performance_history) >= 4:
            recent = self.performance_history[-4:]
            if all(r["mean"] >= th["mean"] for r in recent) and \
               all(r["top"] >= th["top"] for r in recent) and \
               all(r["top3"] >= th["top3"] for r in recent) and \
               all(r["bottom"] >= th["bottom"] for r in recent):
                return True

        # Timeout Safeguard: Auto-advance if the population hits a wall for too many generations
        self.stage_generations += 1
        if self.stage_generations >= self.stage_patience * 6:
            print(f"⏰ Timeout advancement from stage {self.current_stage}")
            return True
            
        return False

    def advance_stage(self):
        """
        Increases the active stage counter by 1 and resets the generation counter to 0.
        """
        if self.current_stage < len(self.config.curriculum_stages) - 1:
            self.current_stage += 1
            self.stage_generations = 0
            print(f"Advancing to stage {self.current_stage}: {self.get_current_stage()['name']}")

# -------- IMAGE PROCESSING 

def _choose_episode_aug(hp: HyperParams):
    """
    Stochastically determine and parameterize input data augmentation configurations 
    (blur, intensity jitter, contrast, and Canny thresholds) at the start of each episode.
    Operates non-invasively; augmentation is fully bypassed if 'aug_prob' is set to 0.

    Args:
        hp (HyperParams): Configuration object where determined options are stashed.
    """
    try:
        # Roll for augmentation activation based on predefined probability configuration
        enable = (random.random() < float(getattr(hp, "aug_prob", 0.0)))
        blur_k = 0
        canny_low = int(hp.canny_low)
        canny_high = int(hp.canny_high)
        b_mul = 1.0
        c_mul = 1.0
        
        if enable:
            # 1) Determine random Gaussian Blur kernel configuration
            choices = getattr(hp, "blur_k_choices", (0,3))
            if isinstance(choices, (list, tuple)) and len(choices) > 0:
                blur_k = int(random.choice(choices))
                if blur_k % 2 == 0 and blur_k != 0:
                    blur_k = blur_k - 1 if blur_k > 1 else 0
            
            # 2) Inject randomized jitter into baseline Canny tracking boundaries
            cj = float(getattr(hp, "canny_jitter_pct", 0.15))
            j_lo = 1.0 + random.uniform(-cj, cj)
            j_hi = 1.0 + random.uniform(-cj, cj)
            canny_low  = max(1, int(round(hp.canny_low  * j_lo)))
            canny_high = max(canny_low+1, int(round(hp.canny_high * j_hi)))
            
            # 3) Generate brightness and contrast scaling factors to simulate light variations
            bj = float(getattr(hp, "brightness_jitter_pct", 0.12))
            cj2 = float(getattr(hp, "contrast_jitter_pct", 0.12))
            b_mul = max(0.5, 1.0 + random.uniform(-bj, bj))
            c_mul = max(0.5, 1.0 + random.uniform(-cj2, cj2))
            
        # Stash calculated profiles back into the configuration storage state
        hp._episode_aug = dict(enable=enable, blur_k=blur_k, canny_low=canny_low, canny_high=canny_high,
                               b_mul=b_mul, c_mul=c_mul)
                               
    except Exception:
        hp._episode_aug = dict(enable=False, blur_k=0, canny_low=int(hp.canny_low), canny_high=int(hp.canny_high),
                               b_mul=1.0, c_mul=1.0)


def _choose_adaptive_crop_ratio(obs: np.ndarray, hp: HyperParams) -> float:
    """
    Dynamically select the optimal image crop ratio based on local edge density 
    and the presence of upcoming diagonal structures (turns or walls).
    Decreases the ratio (looks higher up) when blind, and increases it (focuses closer) 
    when the track layout is clear.

    Args:
        obs (np.ndarray): The raw RGB or grayscale input observation frame.
        hp (HyperParams): Configuration object holding sensory and morphology thresholds.

    Returns:
        float: Normalized crop ratio boundary clamped between strict safety limits.
    """
    try:
        # 1) Preprocess: Enforce Grayscale and Resolution Downscaling
        if obs.ndim == 3 and obs.shape[2] >= 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        small = cv2.resize(gray, (int(getattr(hp, "resize_w", 80)), int(getattr(hp, "resize_h", 60)) ))

        # 2) Baseline Canny: Raw Matrix for Accurate Analytical Counting
        ep_aug = getattr(hp, "_episode_aug", {}) or {}
        t1 = int(ep_aug.get("canny_low",  getattr(hp, "canny_low", 30)))
        t2 = int(ep_aug.get("canny_high", getattr(hp, "canny_high", 100)))
        edges_raw = cv2.Canny(small, t1, t2)
        h, w = edges_raw.shape

        # 3) Hough Canny Matrix: Apply Morphology to Fuse Fragmented Segments
        edges_h = edges_raw.copy()
        if bool(getattr(hp, "use_morph_close", True)):
            k = int(getattr(hp, "morph_kernel", 3))
            if k % 2 == 0:
                k += 1
            iters = int(getattr(hp, "morph_iters", 1))
            ker = np.ones((k, k), np.uint8)

            row_start = int(h * float(getattr(hp, "diag_roi_row_start", 0.60)))
            col_lo = int(w * float(getattr(hp, "diag_roi_col_lo", 0.20)))
            col_hi = int(w * float(getattr(hp, "diag_roi_col_hi", 0.80)))
            col_lo = max(0, min(col_lo, w))
            col_hi = max(col_lo + 1, min(col_hi, w))

            # Restrict morphology expansion exclusively to the active look-ahead ROI window
            sub = edges_h[row_start:, col_lo:col_hi]
            sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, ker, iterations=iters)
            edges_h[row_start:, col_lo:col_hi] = sub

        # 4) Density Evaluation: Calculated Directly from the Raw Edge Frame
        r0 = int(h * float(getattr(hp, "ac_edge_band_lo", 0.65)))
        r1 = int(h * float(getattr(hp, "ac_edge_band_hi", 0.85)))
        r0 = max(0, min(r0, h))
        r1 = max(r0 + 1, min(r1, h))
        roi_den = edges_raw[r0:r1, :]
        density = float(np.mean(roi_den > 0)) if roi_den.size else 0.0

        # 5) Diagonal Line Tracking: Evaluate Split Left/Right ROI Fields
        row_start = int(h * float(getattr(hp, "diag_roi_row_start", 0.60)))
        col_lo = int(w * float(getattr(hp, "diag_roi_col_lo", 0.20)))
        col_hi = int(w * float(getattr(hp, "diag_roi_col_hi", 0.80)))
        col_lo = max(0, min(col_lo, w))
        col_hi = max(col_lo + 1, min(col_hi, w))
        col_mid = (col_lo + col_hi) // 2

        angle_min = float(getattr(hp, "diag_angle_min", getattr(hp, "ac_angle_min", 12.0)))
        angle_max = float(getattr(hp, "diag_angle_max", getattr(hp, "ac_angle_max", 80.0)))
        need_lines = int(getattr(hp, "ac_diag_need_lines", 1))

        def _has_diagonals(roi_edges: np.ndarray) -> bool:
            if roi_edges.size == 0:
                return False
            lines, _ = _houghp_adaptive(
                roi_edges,
                int(getattr(hp, "hough_threshold", 20)),
                hp,
                minLineLength=int(getattr(hp, "diag_min_len", 14)),
                maxLineGap=int(getattr(hp, "diag_max_gap", 8)),
            )
            if lines is None:
                return False
            cnt = 0
            arr = lines.reshape(-1, 4)
            for x1, y1, x2, y2 in arr:
                dx, dy = x2 - x1, y2 - y1
                angle = 90.0 if dx == 0 else abs(np.degrees(np.arctan2(dy, dx)))
                if angle_min <= angle <= angle_max:
                    cnt += 1
                    if cnt >= need_lines:
                        return True
            return False

        # Execute dual-channel validation checks across left and right horizons
        left_roi  = edges_h[row_start:, col_lo:col_mid]
        right_roi = edges_h[row_start:, col_mid:col_hi]
        left_has_diag  = _has_diagonals(left_roi)
        right_has_diag = _has_diagonals(right_roi)
        have_diag = left_has_diag or right_has_diag

        # 6) Adaptive Ratio Optimization 
        ratio = float(getattr(hp, "ac_default_ratio", 0.40))
        thr = float(getattr(hp, "ac_edge_density_thr", 0.10))
        loosen = float(getattr(hp, "ac_loosen_when_blind", -0.08))   
        tighten = float(getattr(hp, "ac_tighten_when_clear", +0.05)) 
        minor = float(getattr(hp, "ac_minor_adjustment", -0.04))

        # Adjust viewpoint based on target line visibility configurations
        if density < thr:
            ratio += (loosen if not have_diag else minor)
        elif density > thr * 1.5 and have_diag:
            ratio += tighten

        # Enforce operational boundary clamps to prevent extreme view clipping
        rmin = float(getattr(hp, "ac_min_ratio", 0.28))
        rmax = float(getattr(hp, "ac_max_ratio", 0.45))
        ratio = float(np.clip(ratio, rmin, rmax))
        return ratio

    except Exception:
        return float(getattr(hp, "ac_default_ratio", 0.40))


# ------- Episode Calibrator 

def reset_episode_calibration(hp):
    """
    Reset and initialize the calibration tracking dictionary profile.
    Must be executed at the absolute beginning of EVERY driving episode/run 
    to wipe stale historical thresholds and data structures.

    Args:
        hp (HyperParams): Configuration object containing the dynamic profile reference.
    """
    # Construct a clean slate calibration dictionary format
    hp._calib = {     
        "done": False, # Flag indicating if the system has completed sampling and locked decisions       
        "frames": 0, # Incremental counter tracking analyzed runtime data frames  
        "frame_budget": int(getattr(hp, "calib_frames", 30)), # Target sample size volume before evaluating environmental decisions
        
        # Historical metrics collection arrays (wiped clean for the new episode)
        "spreads": [],           # Raw grayscale contrast markers (P95 - P5)
        "densities": [],         # Structural edge densities calculated via static limits
        "p_low": [],             # Candidate adaptive lower Canny threshold samples
        "p_high": [],            # Candidate adaptive upper Canny threshold samples
        
        # Frozen output decision map configurations initialized to default/neutral states
        "decisions": {
            "use_clahe": False,
            "auto_canny": False,
            "canny_low": None,
            "canny_high": None,
        }
    }

def _calib_update(gray, hp, resize_w, resize_h, fixed_t1, fixed_t2, blur_k):
    """
    Accumulate image statistics over a fixed frame budget to make robust execution 
    decisions (e.g., enabling CLAHE or adaptive Auto-Canny thresholds).
    Freezes internal configuration flags once the target frame budget is satisfied.

    Args:
        gray (np.ndarray): Input grayscale image array.
        hp (HyperParams): Configuration object storing the calibration dictionary state.
        resize_w (int): Target width for analytical downscaling.
        resize_h (int): Target height for analytical downscaling.
        fixed_t1 (int): Baseline lower threshold for Canny edge evaluation.
        fixed_t2 (int): Baseline upper threshold for Canny edge evaluation.
        blur_k (int): Kernel size multiplier for Gaussian noise reduction.
    """
    # 0) Init calibration state
    if not hasattr(hp, "_calib") or not isinstance(hp._calib, dict):
        reset_episode_calibration(hp)

    cb = hp._calib
    if cb.get("done", False):
        return  # Calibration profile is frozen; skip evaluation

    # 1) Standardize image format and enforce a robust uint8 matrix footprint
    g = gray
    if g.dtype != np.uint8:
        g = np.asarray(g)
        g_min, g_max = float(g.min()), float(g.max())
        if g_max <= 1.0:
            g = (g * 255.0).astype(np.uint8)
        else:
            g = np.clip(g, 0, 255).astype(np.uint8)

    # Calculate global structural contrast spread (rejection of extreme outliner illumination)
    p5  = float(np.percentile(g, 5))
    p95 = float(np.percentile(g, 95))
    spread = p95 - p5
    cb.setdefault("spreads", []).append(spread)

    # 2) Emulate execution workflow via localized downsizing and noise smoothing
    W, H = int(resize_w), int(resize_h)  
    resized = cv2.resize(g, (W, H))
    if blur_k and blur_k > 1:
        if blur_k % 2 == 0:
            blur_k = max(1, blur_k - 1)  # Guarantee an odd kernel geometry
        resized = cv2.GaussianBlur(resized, (blur_k, blur_k), 0)

    # Enforce relational ordering constraints on fixed Canny limits
    t1_fix = int(fixed_t1)
    t2_fix = int(fixed_t2) if int(fixed_t2) > int(fixed_t1) else int(fixed_t1) + 1
    edges = cv2.Canny(resized, threshold1=t1_fix, threshold2=t2_fix, L2gradient=True)

    # Analyze edge densities across both full frame and targeted lower look-ahead band (ROI)
    den_full = float((edges > 0).mean())
    r0 = int(H * float(getattr(hp, "ac_edge_band_lo", 0.65)))
    r1 = int(H * float(getattr(hp, "ac_edge_band_hi", 0.85)))
    r0 = max(0, min(r0, H))
    r1 = max(r0 + 1, min(r1, H))
    roi = edges[r0:r1, :]
    den_band = float((roi > 0).mean()) if roi.size else 0.0

    cb.setdefault("densities_full", []).append(den_full)
    cb.setdefault("densities_band", []).append(den_band)
    cb.setdefault("densities", []).append(den_band) 

    # 3) Gather matching candidate mathematical percentiles via Sobel magnitude profiling
    p1 = int(getattr(hp, "auto_canny_p1", 25))
    p2 = int(getattr(hp, "auto_canny_p2", 75))
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    nz = mag[mag > 0]

    if nz.size > 20:
        v1 = float(np.percentile(nz, p1))
        v2 = float(np.percentile(nz, p2))
    else:
        v1 = float(fixed_t1)
        v2 = float(fixed_t2)

    if v2 <= v1:  # Absolute mathematical separation guard
        v2 = v1 + 1.0
    cb.setdefault("p_low", []).append(v1)
    cb.setdefault("p_high", []).append(v2)

    # 4) Increment step tracker and process compiled decisions once frame budget clears
    cb["frames"] = cb.get("frames", 0) + 1
    if cb["frames"] >= int(cb.get("frame_budget", 10)):
        spread_thresh = float(getattr(hp, "calib_spread_thresh", 45.0))
        den_lo = float(getattr(hp, "calib_edge_density_lo", 0.02))
        den_hi = float(getattr(hp, "calib_edge_density_hi", 0.15))

        # Use robust median values to bypass singular anomalies or noise spikes
        m_spread = float(np.median(cb.get("spreads", [999.0])))
        m_den    = float(np.median(cb.get("densities", [0.0])))
        m_v1     = float(np.median(cb.get("p_low",  [None])))
        m_v2     = float(np.median(cb.get("p_high", [None])))

        # Decision Node A: Enable CLAHE equalizer under flat/low-contrast circumstances
        use_clahe_dec = (m_spread < spread_thresh) and (m_den < den_hi)

        # Decision Node B: Adopt adaptive Auto-Canny if static bounds under-perform or saturate
        auto_canny_dec = (m_den < den_lo) or (m_den > den_hi)

        t1_auto = None
        t2_auto = None
        if auto_canny_dec and (m_v1 is not None) and (m_v2 is not None):
            t1_auto = int(round(m_v1))
            t2_auto = int(round(m_v2))
            if t2_auto <= t1_auto:
                t2_auto = t1_auto + 1
            t1_auto = max(1, min(254, t1_auto))
            t2_auto = max(2, min(255, t2_auto))

        # Pack structured decisions securely back into the profile dictionary
        cb["decisions"] = {
            "use_clahe": use_clahe_dec,
            "auto_canny": auto_canny_dec,
            "canny_low": t1_auto,
            "canny_high": t2_auto,
            "m_spread": m_spread,
            "m_density": m_den,
        }
        cb["done"] = True


# ---- Preprocess 

def _show_roi_debug(obs_world: np.ndarray, hp) -> None:
    """
    Generate and render a synchronized, split-screen diagnostic overlay.
    Displays the raw frame with global thresholds on the left, and the cropped, 
    downscaled execution matrix with re-mapped dynamic ROI boundaries on the right.

    Args:
        obs_world (np.ndarray): The raw sensor/camera frame array.
        hp (HyperParams): Configuration object containing runtime tracking profiles.
    """
    # --- sanity ---
    if obs_world is None or obs_world.size == 0:
        print("[overlay] skipped: empty frame")
        return

    # Log initial entry confirmation only once to prevent console spamming
    if not getattr(hp, "_overlay_seen", False):
        print("[overlay] first call OK")
        setattr(hp, "_overlay_seen", True)

    # Left Canvas: Global World Feed 
    if obs_world.ndim == 2: # Convert grayscale tracking matrix to BGR for color telemetry text
        left = cv2.cvtColor(obs_world, cv2.COLOR_GRAY2BGR)
    else:
        left = cv2.cvtColor(obs_world, cv2.COLOR_RGB2BGR)
    h0, w0 = left.shape[:2]

    # Draw sky/horizon baseline crop marker
    rt = getattr(hp, "_runtime", {}) or {}
    c = float(rt.get("crop_ratio", 0.0))
    y_crop = int(h0 * c)
    cv2.line(left, (0, y_crop), (w0-1, y_crop), (0, 255, 255), 1)
    cv2.putText(left, f"crop c={c:.2f}", (8, max(12, y_crop-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

    # Define targets to track across the dynamic segmentation loops
    items = [
        ("diag", float(getattr(hp, "diag_roi_row_start", 0.58))),
        ("side", float(getattr(hp, "side_line_row_start", 0.60))),
        ("flat", float(getattr(hp, "flat_roi_row_start", 0.66))),
    ]

    # Right Canvas: Post-Crop Execution Preview
    start = y_crop
    cropped = obs_world[start:, :] if obs_world.ndim == 2 else obs_world[start:, :, :]
    if cropped.size == 0:
        print("[overlay] cropped frame empty; increase ac_max_ratio?")
        return

    # Emulate the model's localized resolution resizing
    gray = cropped if cropped.ndim == 2 else cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    H = int(getattr(hp, "resize_h", 60)); W = int(getattr(hp, "resize_w", 80))
    small = cv2.resize(gray, (W, H))
    right = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    def _row_start_eff(alpha_world: float) -> float:
        # Scale world row ratios into post-cropped relative coordinates
        c_ = float((getattr(hp, "_runtime", {}) or {}).get("crop_ratio", 0.0))
        denom = max(1e-6, 1.0 - c_)
        r = (alpha_world - c_) / denom
        return float(np.clip(r, 0.0, 0.999999))

    # Draw Synchronized Visual Markers
    for name, a in items:
        # Plot markers on the absolute world frame
        yw = int(h0 * a)
        cv2.line(left, (0, yw), (w0-1, yw), (255,128,0), 1)
        cv2.putText(left, f"{name} a={a:.2f}", (8, min(h0-6, yw+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,128,0), 1, cv2.LINE_AA)

        # Compute and plot markers on the shifted/cropped frame
        use_world = bool(getattr(hp, "roi_world_coords", True))
        r = _row_start_eff(a) if use_world else a
        yr = int(H * r)
        cv2.line(right, (0, yr), (W-1, yr), (255,128,0), 1)
        rec = c + (1.0 - c) * r
        cv2.putText(right, f"{name}: r={r:.3f} (rec={rec:.2f})", (4, max(12, yr-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,128,0), 1, cv2.LINE_AA)

    # Canvas Compiling
    pad = 12
    Hc = max(h0, H); Wc = w0 + pad + W
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    canvas[:h0, :w0] = cv2.resize(left, (w0, h0))
    canvas[:H, w0+pad:w0+pad+W] = right

    # Robust Output Routing & Fail-safe Handling
    mode = str(getattr(hp, "debug_overlay_output", "window")).lower()
    try:
        if mode == "window":
            cv2.imshow("ROI debug  (left: world | right: post-crop)", canvas)
            cv2.waitKey(1)
        elif mode == "images":
            outdir = getattr(hp, "debug_overlay_dir", "roi_debug")
            os.makedirs(outdir, exist_ok=True)
            fi = int(getattr(hp, "_frame_i", 0))
            cv2.imwrite(os.path.join(outdir, f"roi_{fi:06d}.png"), canvas)
        elif mode == "video":
            vw = getattr(hp, "_vw", None)
            if vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter("roi_debug.mp4", fourcc, 20, (canvas.shape[1], canvas.shape[0]))
                setattr(hp, "_vw", vw)
            vw.write(canvas)
        else:
            # Fallback output routing block
            outdir = getattr(hp, "debug_overlay_dir", "roi_debug")
            os.makedirs(outdir, exist_ok=True)
            fi = int(getattr(hp, "_frame_i", 0))
            cv2.imwrite(os.path.join(outdir, f"roi_{fi:06d}.png"), canvas)
    except Exception as e:
        # Automated fallback protection to save data frames if display driver is headless
        outdir = getattr(hp, "debug_overlay_dir", "roi_debug")
        os.makedirs(outdir, exist_ok=True)
        fi = int(getattr(hp, "_frame_i", 0))
        cv2.imwrite(os.path.join(outdir, f"roi_{fi:06d}.png"), canvas)
        if not getattr(hp, "_overlay_warned", False):
            print(f"[overlay] window mode failed, saved PNG instead. err={e}")
            setattr(hp, "_overlay_warned", True)


def preprocess(obs: np.ndarray, hp: Any) -> np.ndarray:
    """Preprocesses raw environment observations into optimized edge-detected arrays.
    
    This pipeline transforms a high-resolution color image into a low-resolution,
    normalized single-channel edge map. It applies adaptive cropping, grayscale 
    conversion, contrast enhancement (CLAHE), dynamic Canny thresholding, shadow 
    suppression, and morphological operations to isolate structural road layout features 
    for downstream Neural Network processing.

    Args:
        obs: Raw input frame image array, typically shape (H, W, 3) RGB or (H, W) grayscale.
        hp: Hyperparameter configuration object containing structural parameters, 
            thresholds, and dynamic runtime stats (`_runtime`, `_calib`, `_episode_aug`).

    Returns:
        A normalized float32 array shaped (1, Target_H, Target_W) with pixel values 
        scaled between [0.0, 1.0].
    """
    # --- Phase 1: Adaptive Field-of-View Cropping ---
    crop_ratio = float(getattr(hp, "crop_top_ratio", 0.0))
    if bool(getattr(hp, "adaptive_crop_enabled", False)):
        crop_ratio = _choose_adaptive_crop_ratio(obs, hp)
    crop_ratio = float(np.clip(crop_ratio, 0.0, 0.95))

    obs_world = obs 

    # Cache calculated crop settings for tracking or telemetry purposes
    rt = dict(getattr(hp, "_runtime", {}) or {})
    rt["crop_ratio"] = crop_ratio
    setattr(hp, "_runtime", rt)

    h0 = int(obs.shape[0])
    start_row = int(h0 * crop_ratio)
    cropped = obs[start_row:]
    
    # --- Phase 2: Grayscale Conversion ---
    if cropped.ndim == 3 and cropped.shape[2] >= 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    else:
        gray = cropped

    # --- Phase 3: Dynamic Calibration Assessment ---
    base_t1 = int(getattr(hp, "canny_low", 30))
    base_t2 = int(getattr(hp, "canny_high", 100))
    blur_k  = int(getattr(hp, "blur_ksize", 0))
    _calib_update(gray, hp, int(getattr(hp, "resize_w", 80)), int(getattr(hp, "resize_h", 60)), base_t1, base_t2, blur_k)

    cb = getattr(hp, "_calib", None)
    use_clahe_eff  = bool(getattr(hp, "use_clahe", False))
    auto_canny_eff = bool(getattr(hp, "auto_canny", False))
    t1_auto = t2_auto = None
    
    if isinstance(cb, dict) and cb.get("done", False):
        use_clahe_eff  = use_clahe_eff or bool(cb["decisions"].get("use_clahe", False))
        auto_canny_eff = auto_canny_eff or bool(cb["decisions"].get("auto_canny", False))
        t1_auto = cb["decisions"].get("canny_low", None)
        t2_auto = cb["decisions"].get("canny_high", None)

    # --- Phase 4: Contrast Limited Adaptive Histogram Equalization (CLAHE) ---
    if use_clahe_eff:
        clahe = cv2.createCLAHE(
            clipLimit=float(getattr(hp, "clahe_clip", 2.0)),
            tileGridSize=(int(getattr(hp, "clahe_grid", 8)), int(getattr(hp, "clahe_grid", 8)))
        )
        gray = clahe.apply(gray)

    # --- Phase 5: Dimensionality Reduction (Resize) ---
    W = int(getattr(hp, "resize_w", 80))
    H = int(getattr(hp, "resize_h", 60))
    resized = cv2.resize(gray, (W, H))

    # --- Phase 6: Noise Mitigation (Gaussian Blur) ---
    bk = int(getattr(hp, "blur_ksize", 0))
    ep = getattr(hp, "_episode_aug", None)
    if isinstance(ep, dict) and ep.get("enable", False):
        bk = int(ep.get("blur_k", bk))
    if bk > 1:
        if bk % 2 == 0:
            bk -= 1  # Kernel sizes must remain odd for Gaussian filters
        resized = cv2.GaussianBlur(resized, (bk, bk), 0)

    # --- Phase 7: Edge Feature Extraction (Dynamic Canny) ---
    t1, t2 = base_t1, base_t2

    # Fallback onto on-the-fly gradient magnitude percentiles if calibration is active but incomplete
    if auto_canny_eff and not (t1_auto is None or t2_auto is None):
        t1, t2 = int(t1_auto), int(t2_auto)
    elif auto_canny_eff:
        p1 = int(getattr(hp, "auto_canny_p1", 25))
        p2 = int(getattr(hp, "auto_canny_p2", 75))

        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        nz = mag[mag > 0]

        if nz.size > 0:
            lo = float(np.percentile(nz, p1))
            hi = float(np.percentile(nz, p2))
            t1, t2 = int(lo), max(int(hi), int(lo) + 1)
        else:
            t1, t2 = base_t1, base_t2

    # Apply external evaluation/episode parameter overrides if present
    if isinstance(ep, dict) and ep.get("enable", False):
        t1 = int(ep.get("canny_low", t1))
        t2 = int(ep.get("canny_high", t2))
    if t2 <= t1:
        t2 = t1 + 1

    # Apply structural scaling derived from Genetic Algorithm behavioral parameters
    scale = float(getattr(hp, "canny_scale", 1.0))
    if not np.isfinite(scale):
        scale = 1.0
    t1 = int(np.clip(round(t1 * scale), 1, 254))
    t2 = int(np.clip(round(t2 * scale), t1 + 1, 255))

    edges = cv2.Canny(resized, threshold1=t1, threshold2=t2, L2gradient=True)

    # --- Phase 8: Illumination Artifact / Shadow Suppression ---
    if bool(getattr(hp, "shadow_suppression", True)) and cropped.ndim == 3 and cropped.shape[2] >= 3:
        edge_density = float((edges > 0).mean())
        den_hi = float(getattr(hp, "calib_edge_density_hi", 0.25))

        # Suppress features if spatial noise bounds indicate heavy over-segmentation in dark shadows
        if edge_density > den_hi:
            hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
            V = hsv[..., 2]
            Vr = cv2.resize(V, (W, H))

            p1 = int(getattr(hp, "auto_canny_p1", 25))
            v_thr = np.percentile(Vr, p1)

            shadow = Vr < v_thr
            edges[shadow] = 0

    # --- Phase 9: Morphological Geometry Closing ---
    if bool(getattr(hp, "use_morph_close", True)) and bool(getattr(hp, "morph_in_preprocess", False)):
        mk = int(getattr(hp, "morph_kernel", 3))
        if mk % 2 == 0:
            mk += 1
        iters = int(getattr(hp, "morph_iters", 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((mk, mk), np.uint8), iterations=iters)

    # --- Phase 10: Instrumentation / Telemetry Overlays ---
    fi = int(getattr(hp, "_frame_i", 0))
    setattr(hp, "_frame_i", fi + 1)
    if bool(getattr(hp, "debug_overlay", True)) and (fi % int(getattr(hp, "debug_overlay_every_n", 5)) == 0):
        _show_roi_debug(obs_world, hp)

    # --- Phase 11: Range Normalization & Tensor Reshaping ---
    return (edges.astype(np.float32) / 255.0).reshape((1, H, W))

def _houghp_adaptive(img, base_thr: int, hp: HyperParams, *, minLineLength: int, maxLineGap: int, min_thr: int = None, step: int = None):
    """
    Adaptive wrapper around cv2.HoughLinesP that progressively lowers the detection 
    threshold if no lines are found at the baseline sensitivity level.

    Args:
        img (np.ndarray): The input binary/edge image region to scan.
        base_thr (int): The initial/preferred voting threshold for line detection.
        hp (HyperParams): Configuration object containing fallback limit structures.
        minLineLength (int): Minimum length of line segment to be accepted.
        maxLineGap (int): Maximum allowed gap between points on the same line to link them.
        min_thr (int, optional): The lowest floor threshold to drop down to. Defaults to None.
        step (int, optional): The decrement step size for each adaptive iteration. Defaults to None.

    Returns:
        Tuple[Optional[np.ndarray], int]: A tuple containing the detected lines array (or None) 
                                          and the final threshold integer used to find them.
    """

    try:
        mn = int(getattr(hp, "hough_min_threshold", 12)) if min_thr is None else int(min_thr)
        st = int(getattr(hp, "hough_adapt_step", 3)) if step is None else int(step)
        thr = int(base_thr)

        # Isolated execution wrapper for the core OpenCV Hough operator
        def call(th):
            return cv2.HoughLinesP(img, 1, np.pi/180, threshold=max(1, th), minLineLength=minLineLength, maxLineGap=maxLineGap)
        
        # First attempt: Try to catch strong, confident lines using the baseline threshold
        lines = call(thr)
        if lines is not None:
            return lines, thr
        
        # Fallback loop: Progressively lower the threshold to recover faint or fragmented lines
        for th in range(max(1, thr - st), max(0, mn - 1), -st):
            lines = call(th)
            if lines is not None:
                return lines, th
        return None, thr
    except Exception:
        # Safe fallback failure state to guard against unexpected runtime shape or type mutations
        return None, base_thr

# ----- LANE / TURN DETECTION

def _auto_canny_from_sample(gray_roi: np.ndarray, scale: float = 1.0) -> Tuple[int, int]:
    """
    Dynamically calculate adaptive low and high thresholds for the Canny edge detector 
    based on the gradient intensity distribution (percentiles) of a sample ROI.

    Args:
        gray_roi (np.ndarray): The grayscale sample region of interest to analyze.
        scale (float): A tuning multiplier to scale the calculated thresholds. Defaults to 1.0.

    Returns:
        Tuple[int, int]: Adaptive (low, high) threshold bounds for Canny edge detection.
    """
    
    # Compute Sobel gradients to build a texture-resistant edge magnitude map
    gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    
    # Extract structural contrast markers using statistical percentiles
    p20 = float(np.percentile(mag, 20))
    p80 = float(np.percentile(mag, 80))

    # Map percentiles to standard Canny bounds with rigid bottom-floor safety limits
    lo, hi = int(max(5.0, p20 * 0.9 * scale)), int(max(10.0, p80 * 1.1 * scale))

    # Defensive guard to guarantee a valid mathematical span between bounds
    if lo >= hi: hi = lo + 5
    return lo, hi

def detect_side_lines(edge_image: np.ndarray, hp: HyperParams) -> Tuple[bool, bool]:
    """
    Check for the physical presence of lane markings on both the left and right 
    sides of the immediate look-ahead field (lower ROI).

    Args:
        edge_image (np.ndarray): The edge map 
        hp (HyperParams): Configuration object containing coordinates and pixel counters.

    Returns:
        Tuple[bool, bool]: 
            - has_left (bool): True if the left side line is visible/detected.
            - has_right (bool): True if the right side line is visible/detected.
    """
    img = (edge_image[0] * 255).astype(np.uint8)
    h, w = img.shape

    # Segment the lower region of the image and split it into Left and Right halves
    r0 = int(h * hp.side_line_row_start)
    ll = img[r0:, : w // 2]
    lr = img[r0:, w // 2:]

    # Load parameters for pixel intensity validation and minimum density counts
    thr  = int(hp.side_line_pixel_threshold)
    minc = int(hp.side_line_min_count)

    # Verify line presence by ensuring valid edge pixel volume exceeds thresholds
    has_left  = np.sum(ll > thr) > minc
    has_right = np.sum(lr > thr) > minc

    return has_left, has_right


def detect_dashed_centerline(edge_image, hp):
    """
    Detect and track a dashed centerline inside a targeted central region (ROI).
    Identifies standalone dash segments using connected components, fits a trajectory line, 
    and measures the vehicle's lateral offset relative to it.

    Args:
        edge_image (np.ndarray): The edge map 
        hp (HyperParams): Configuration object containing geometric thresholds.

    Returns:
        Tuple[float, float]: 
            - score (float): Reliability/confidence score of the detection (0.0 to 1.0).
            - offset_norm (float): Normalized lateral distance from the center (-1.0 to 1.0).
    """
    img = edge_image[0] if getattr(edge_image, "ndim", 2) == 3 else edge_image
    img = (img > 0.5).astype(np.uint8)
    h, w = img.shape

    # Isolate the central lane-seeking field of view (ROI)
    r0 = int(h * float(getattr(hp, "dash_row0", 0.55)))
    r1 = int(h * float(getattr(hp, "dash_row1", 0.95)))
    c0 = int(w * float(getattr(hp, "dash_col0", 0.35)))
    c1 = int(w * float(getattr(hp, "dash_col1", 0.65)))

    roi = img[r0:r1, c0:c1]
    if roi.size == 0:
        return 0.0, 0.0
    
    # Clean up fragmented edges and identify individual standalone shapes (components)
    roi = cv2.morphologyEx(roi*255, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1) > 0
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(roi.astype(np.uint8), 8)

    # Establish bounding box and spatial aspect ratio criteria for real dashes
    H, W = roi.shape
    min_h = max(4, int(H*0.02)); max_h = max(8, int(H*0.12))
    max_w = max(3, int(W*0.12))
    ar_min = 1.4
    xs, ys = [], []
    for i in range(1, num):
        x,y,ww,hh,area = stats[i]

        # Enforce structural filtering (reject shadows, walls, and horizontal noise)
        if hh < min_h or hh > max_h:  continue
        if ww > max_w:                continue
        if hh / max(1, ww) < ar_min:  continue

        # Remap localized component centers back to global frame coordinates
        cx, cy = centroids[i]
        xs.append(cx + c0); ys.append(cy + r0)

    # An acceptable trace line requires a minimum of 4 distinct dash samples
    if len(xs) < 4:
        return 0.0, 0.0
    xs = np.asarray(xs, np.float32); ys = np.asarray(ys, np.float32)

    # Perform standard linear regression (least squares fit) across valid centroids
    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]

    # Evaluate linearity using R-squared metric
    pred = a*ys + b
    ss_res = float(np.sum((xs - pred)**2))
    ss_tot = float(np.sum((xs - xs.mean())**2) + 1e-6)
    r2 = 1.0 - ss_res/ss_tot

    # Analyze mathematical distance gaps between successive dashes
    order = np.argsort(ys)
    gaps = np.diff(ys[order])
    if gaps.size == 0:
        return 0.0, 0.0

    # Compute Coefficient of Variation (CV) to test spatial uniformity/periodicity
    cv = float(np.std(gaps) / (float(np.mean(gaps)) + 1e-6))

    # Project the fitted line down to a close-range look-ahead reference index
    y_ref = float(h * 0.85)
    x_line = float(a*y_ref + b)

    # Normalize final vehicle lateral error/offset command
    offset_norm = float((x_line - (w*0.5)) / (w*0.5))

    # Fuse tracking attributes into a comprehensive multi-factored confidence score
    seg_term  = float(np.tanh((len(xs)-3)/4))
    r2_term   = max(0.0, min(1.0, float(r2)))
    cv_term   = float(np.exp(-min(cv, 2.0)))
    score = float(0.45*r2_term + 0.35*cv_term + 0.20*seg_term)
    return score, offset_norm

def detect_diagonal_angle(edge_image: np.ndarray, hp: HyperParams) -> Optional[float]:
    """
    Detect if there is a dangerous diagonal line (like a track wall or a sharp apex) 
    within a specific look-ahead window (ROI).

    Args:
        edge_image (np.ndarray): The edge map 
        hp (HyperParams): Configuration object containing thresholds and ROI coordinates.

    Returns:
        Optional[float]: The angle of the detected diagonal line in degrees, 
                         or None if no dangerous diagonal lines are found.
    """
    img = (edge_image[0] * 255).astype(np.uint8)
    h, w = img.shape

    # Define the custom boundaries for the diagonal look-ahead area (ROI)
    c0 = int(w * hp.diag_roi_col_lo)
    c1 = int(w * hp.diag_roi_col_hi)
    roi = img[int(h * hp.diag_roi_row_start):, c0:c1]
    lines, _thr_used = _houghp_adaptive(roi, hp.hough_threshold, hp, minLineLength=hp.diag_min_len, maxLineGap=hp.diag_max_gap)
    if lines is None:
        return None
    
    # Iterate through detected lines to calculate and validate their slope/angle
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Return the angle immediately if it falls within the critical diagonal safety thresholds
        if hp.diag_angle_min <= abs(angle) <= hp.diag_angle_max:
            return float(angle)
    return None

def detect_flat_lines(edge_image: np.ndarray, hp: HyperParams) -> bool:
    """
    Detect if there are horizontal or flat lines (such as straight track markings 
    or horizontal shadow artifacts) within a specific central look-ahead window (ROI).

    Args:
        edge_image (np.ndarray): The edge map 
        hp (HyperParams): Configuration object containing thresholds and ROI coordinates.

    Returns:
        bool: True if at least one flat/horizontal line is detected, False otherwise.
    """
    img = (edge_image[0] * 255).astype(np.uint8)
    h, w = img.shape

    # Isolate the specific region of the road (ROI)
    roi = img[int(h * hp.flat_roi_row_start): h, int(w * hp.flat_roi_col_lo): int(w * hp.flat_roi_col_hi)]

    # Run adaptive Hough transform to extract line segments inside the ROI
    lines, _thr_used = _houghp_adaptive(roi, hp.hough_threshold, hp, minLineLength=hp.flat_min_len, maxLineGap=hp.flat_max_gap)
    if lines is None:
        return False
    
    # Iterate through the lines to evaluate their orientation relative to the horizon
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Check if the line is horizontal (near 0 or 180 degrees) based on the threshold
        flat = (abs(angle) < hp.straight_angle_deg) or (abs(abs(angle) - 180) < hp.straight_angle_deg)
        if flat:
            return True
    return False


def detect_symmetric_midline(edge_image: np.ndarray, hp: HyperParams):
    """
    Dynamic symmetric lane detector:
    - no fixed left/right x windows
    - finds lane-edge candidates from lower-ROI column histograms
    - tracks left/right edge points per row around dynamic seeds
    - fits x = a*y + b for both sides
    - estimates centerline and confidence
    """
    if edge_image is None:
        return 0.0, 0.0, 0.0, {"ok": False, "reason": "no_image"}

    img = edge_image[0] if getattr(edge_image, "ndim", 2) == 3 else edge_image
    img = (img > 0.5).astype(np.uint8)

    h, w = img.shape
    r0 = int(h * float(getattr(hp, "sym_row0", 0.58)))
    r1 = int(h * float(getattr(hp, "sym_row1", 0.95)))
    r0 = max(0, min(r0, h))
    r1 = max(r0 + 1, min(r1, h))

    roi = img[r0:r1, :]
    if roi.size == 0:
        return 0.0, 0.0, 0.0, {"ok": False, "reason": "empty_roi"}

    # Light cleanup
    roi = cv2.morphologyEx(
        roi * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    roi = (roi > 0).astype(np.uint8)

    # 1) Column histogram in lower ROI
    col_sum = roi.sum(axis=0).astype(np.float32)
    if col_sum.max() <= 0:
        return 0.0, 0.0, 0.0, {"ok": False, "reason": "no_edges"}

    # Smooth histogram
    k = max(5, int(w * 0.05))
    if k % 2 == 0:
        k += 1
    col_sum_s = cv2.GaussianBlur(col_sum.reshape(1, -1), (k, 1), 0).reshape(-1)

    cx = w // 2
    margin = max(6, int(w * 0.06))

    left_hist = col_sum_s[:max(1, cx - margin)]
    right_hist = col_sum_s[min(w - 1, cx + margin):]

    if left_hist.size < 5 or right_hist.size < 5:
        return 0.0, 0.0, 0.0, {"ok": False, "reason": "narrow_hist"}

    left_seed = int(np.argmax(left_hist))
    right_seed = int(np.argmax(right_hist)) + (cx + margin)

    # Weak histogram -> bail out
    if col_sum_s[left_seed] < 3 or col_sum_s[right_seed] < 3:
        return 0.0, 0.0, 0.0, {"ok": False, "reason": "weak_peaks"}

    # 2) Trace left/right edge points row by row around dynamic seeds
    search_half = max(5, int(w * 0.08))
    left_pts = []
    right_pts = []

    cur_left = left_seed
    cur_right = right_seed

    for yy in range(roi.shape[0] - 1, -1, -1):
        row = roi[yy]
        nz = np.flatnonzero(row)
        if nz.size == 0:
            continue

        # left candidate near current left
        lx0 = max(0, cur_left - search_half)
        lx1 = min(w, cur_left + search_half + 1)
        left_cands = nz[(nz >= lx0) & (nz < lx1)]
        if left_cands.size > 0:
            lx = int(left_cands[np.argmin(np.abs(left_cands - cur_left))])
            left_pts.append((yy + r0, lx))
            cur_left = lx

        # right candidate near current right
        rx0 = max(0, cur_right - search_half)
        rx1 = min(w, cur_right + search_half + 1)
        right_cands = nz[(nz >= rx0) & (nz < rx1)]
        if right_cands.size > 0:
            rx = int(right_cands[np.argmin(np.abs(right_cands - cur_right))])
            right_pts.append((yy + r0, rx))
            cur_right = rx

    if len(left_pts) < 12 or len(right_pts) < 12:
        return 0.0, 0.0, 0.0, {
            "ok": False,
            "reason": "few_points",
            "n_left": len(left_pts),
            "n_right": len(right_pts),
        }

    def _fit_line(pts):
        ys = np.array([p[0] for p in pts], dtype=np.float32)
        xs = np.array([p[1] for p in pts], dtype=np.float32)

        A = np.vstack([ys, np.ones_like(ys)]).T
        a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
        pred = a * ys + b
        ss_res = float(np.sum((xs - pred) ** 2))
        ss_tot = float(np.sum((xs - xs.mean()) ** 2) + 1e-6)
        r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        return {
            "a": float(a),
            "b": float(b),
            "r2": float(r2),
            "n": int(len(pts)),
            "ys": ys,
            "xs": xs,
        }

    L = _fit_line(left_pts)
    R = _fit_line(right_pts)

    # 3) Centerline
    y_ref = float(h * 0.85)
    xL = L["a"] * y_ref + L["b"]
    xR = R["a"] * y_ref + R["b"]

    width = float(xR - xL)
    width_norm = width / max(1.0, float(w))

    # Reject obvious nonsense only
    if width <= 4 or width_norm < 0.10 or width_norm > 0.95:
        return 0.0, 0.0, 0.0, {
            "ok": False,
            "reason": "bad_width",
            "width_norm": width_norm,
        }

    # Runtime EMA width prior
    rt = dict(getattr(hp, "_runtime", {}) or {})
    prev_width = rt.get("sym_width_ema", None)
    if prev_width is None:
        width_consistency = 1.0
        rt["sym_width_ema"] = width_norm
    else:
        prev_width = float(prev_width)
        rel_err = abs(width_norm - prev_width) / max(1e-3, prev_width)
        width_consistency = float(np.exp(-3.0 * rel_err))
        rt["sym_width_ema"] = 0.85 * prev_width + 0.15 * width_norm
    setattr(hp, "_runtime", rt)

    denom = abs(L["a"]) + abs(R["a"]) + 1e-6
    slope_sym = 1.0 - min(1.0, abs(L["a"] + R["a"]) / denom)

    a_c = 0.5 * (L["a"] + R["a"])
    b_c = 0.5 * (L["b"] + R["b"])
    x_mid = a_c * y_ref + b_c

    offset_norm = float((x_mid - (w * 0.5)) / (w * 0.5))
    heading_err = float(a_c)

    support_term = float(np.tanh((min(L["n"], R["n"]) - 10) / 12.0))
    fit_term = float(min(L["r2"], R["r2"]))

    conf_sym = float(
        0.30 * fit_term +
        0.25 * slope_sym +
        0.25 * support_term +
        0.20 * width_consistency
    )
    conf_sym = float(np.clip(conf_sym, 0.0, 1.0))

    dbg = {
        "ok": True,
        "left_seed": left_seed,
        "right_seed": right_seed,
        "n_left": L["n"],
        "n_right": R["n"],
        "r2L": L["r2"],
        "r2R": R["r2"],
        "width_px": width,
        "width_norm": width_norm,
        "xL": float(xL),
        "xR": float(xR),
        "x_mid": float(x_mid),
        "y_ref": float(y_ref),
        "width_consistency": width_consistency,
        "slope_sym": slope_sym,
        "offset_norm": offset_norm,
        "heading_err": heading_err,
    }
    return conf_sym, offset_norm, heading_err, dbg

# ---------- TURN SIGNALS

def detect_turn_signals(edge_image: np.ndarray, hp, debug: bool = False) -> Dict[str, Any]:
    """
    Multi-ROI, weighted turn detector optimized to mitigate shadows and wall artifacts.
    Uses Hough line tracking for immediate visibility (NEAR) and orientation-filtered 
    spatial density analysis for look-ahead fields (MID/FAR).
    """

    def debug_print(msg: str):
        if debug:
            print(f"[TURN_DEBUG] {msg}")

    # --------------------- helpers ---------------------
    def _as_uint8(img_like: np.ndarray) -> np.ndarray:
        # Enforce safe 2D array dimensions and uint8 normalization

        arr = img_like
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.dtype == np.uint8:
            return arr
        m = float(np.max(arr)) if arr.size else 0.0
        return (np.clip(arr, 0, 255).astype(np.uint8) if m > 1.5
                else (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    

    def _morph_close(block: np.ndarray) -> np.ndarray:
        # Fill spatial gaps along fragmented edge lines to boost Hough performance

        if not getattr(hp, "use_morph_close", True) or block.size == 0:
            return block
        k = int(getattr(hp, "morph_kernel", 3))
        if k % 2 == 0: k += 1
        iters = int(getattr(hp, "morph_iters", 1))
        return cv2.morphologyEx(block, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=iters)

    def _count_diags_weighted(block: np.ndarray, *, min_len: int) -> Tuple[float, int, int]:
        # Quantify diagonal tracking lines using length, center proximity, and angular weights

        if block.size == 0:
            return 0.0, 0, 0
        lines, _ = _houghp_adaptive(
            block,
            int(getattr(hp, "hough_threshold", 20)), hp,
            minLineLength=int(min_len),
            maxLineGap=int(getattr(hp, "diag_max_gap", 8))
        )
        total_weight, line_count, strong_lines = 0.0, 0, 0
        a_min = float(getattr(hp, "diag_angle_min", 12.0))
        a_max = float(getattr(hp, "diag_angle_max", 75.0))  
        optimal = 0.5 * (a_min + a_max)
        half_range = max(1e-6, 0.5 * (a_max - a_min))
        bc = 0.5 * block.shape[1]
        if lines is not None:
            arr = lines.reshape(-1, 4)
            for (x1, y1, x2, y2) in arr:
                ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if not (a_min <= ang <= a_max):
                    continue
                line_count += 1
                length = float(np.hypot(x2 - x1, y2 - y1))
                ang_w = 1.0 - abs(ang - optimal) / half_range
                ang_w = float(np.clip(ang_w, 0.1, 1.0))
                len_w = min(1.0, length / max(1.0, min_len * 1.2))
                len_w = max(0.3, len_w)
                cx = 0.5 * (x1 + x2)
                pos_w = 1.0 - abs(cx - bc) / max(1.0, bc)  
                pos_w = float(np.clip(pos_w, 0.2, 1.0))
                wsum = ang_w * len_w * pos_w
                total_weight += wsum
                if wsum > 0.4:
                    strong_lines += 1
        return total_weight, line_count, strong_lines

    # -------- image & gradients 

    img = _as_uint8(edge_image)
    h, w = img.shape
    debug_print(f"Input: {h}x{w}, edges={(img>0).sum()}")

    # Generate edge orientation map via Sobel operators to filter out horizontal noise
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    ang_full = np.abs(np.degrees(np.arctan2(gy, gx)))  # 0..180

    # --------- dynamic split 
    split = w // 2
    try:
        sconf, soff, shead, sdbg = detect_symmetric_midline(edge_image, hp)
        sconf = float(sconf); soff = float(soff)
    except Exception:
        # Fallback to column-wise center of mass calculation if midline detection fails
        col = img.mean(axis=0).astype(np.float32)
        sconf, soff = 0.0, 0.0
        if col.sum() > 0:
            x = np.arange(w, dtype=np.float32)
            tot = float(col.sum()) + 1e-6
            x_cm = float((x * col).sum() / tot)
            soff = 2.0 * (x_cm / max(1.0, (w - 1))) - 1.0
            left_sum, right_sum = col[:w//2].sum(), col[w//2:].sum()
            total = left_sum + right_sum
            if total > 0:
                sconf = float(min(1.0, abs(left_sum - right_sum) / total * 3.0))
    if sconf > float(getattr(hp, "sym_conf_thr", 0.40)):
        k = float(getattr(hp, "turn_split_gain", 0.30))
        split = int(np.clip((w * 0.5) + (w * 0.5) * k * soff, 0.25 * w, 0.75 * w))

    # ---------- ROI bands 
    r_near0 = int(h * max(0.70, float(getattr(hp, "turninfo_roi_start", 0.65))))
    r_mid0  = int(h * float(getattr(hp, "turn_mid_row_start", 0.50)))
    r_far0  = int(h * float(getattr(hp, "turn_far_row_start", 0.35)))

    min_roi = 5
    r_far0  = int(np.clip(r_far0, 0, h - 3 * min_roi))
    r_mid0  = int(np.clip(max(r_mid0, r_far0 + min_roi), 0, h - 2 * min_roi))
    r_near0 = int(np.clip(max(r_near0, r_mid0 + min_roi), 0, h - min_roi))

    near_roi = img[r_near0:, :]
    mid_roi  = img[r_mid0:r_near0, :]
    far_roi  = img[r_far0:r_mid0, :]

    ang_mid = ang_full[r_mid0:r_near0, :]
    ang_far = ang_full[r_far0:r_mid0, :]

    # Early exit out if the immediate look-ahead field (NEAR) is blind
    if float(np.mean(near_roi > 0)) < float(getattr(hp, "edge_bin_thresh", 0.10)):
        return {
            "lane_confidence": 0.0, "turn_intensity": 0.0, "turn_direction": 0.0,
            "direction_confidence": 0.0, "overall_confidence": 0.0,
            "edges_total": int((near_roi > 0).sum()), "split_px": int(split), "direction": None,
            "lane_conf": 0.0, "dir_conf": 0.0, "overall_conf": 0.0,
            "mid_density": 0.0, "far_density": 0.0, "near_strength": 0.0,
        }

    # --------- NEAR Zone: Hough Tracking with Closing
    left_n  = _morph_close(near_roi[:, :split])
    right_n = _morph_close(near_roi[:, split:])

    Hn, Wn = near_roi.shape[:2]
    side_width = max(split, w - split)
    adapt_min = max(int(getattr(hp, "diag_min_len", 8)),
                    int(0.15 * min(Hn, side_width)))
    Lw, Lcnt, _ = _count_diags_weighted(left_n,  min_len=adapt_min)
    Rw, Rcnt, _ = _count_diags_weighted(right_n, min_len=adapt_min)

    near_strength = float(Lw + Rw)
    near_dir_norm = float((Rw - Lw) / max(1e-6, (Rw + Lw)))  # [-1,1], +→Right

    # ------- MID/FAR Zones: Orientation-Filtered Density
    dens_a_min = float(getattr(hp, "dens_angle_min", 12.0))
    dens_a_max = float(getattr(hp, "dens_angle_max", 78.0))
    margin_frac = float(getattr(hp, "road_interior_margin_frac", 0.12))
    c0 = int(margin_frac * w)
    c1 = int((1.0 - margin_frac) * w)

    def _dens_oriented(block: np.ndarray, ang_block: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates the pixel density of a specific orientation inside an image block.

        It filters pixels based on a target angle range and splits the region into 
        left, right, and total density segments based on a vertical split line.

        Args:
        block: The source image pixel intensity array.
        ang_block: The corresponding array containing pixel gradient angles.

        Returns:
        A tuple containing left density (dl), right density (dr), and total density (dt).
        """
        if block.size == 0:
            return 0.0, 0.0, 0.0
        B = block[:, c0:c1]
        A = ang_block[:, c0:c1]
        mask = (B > 0) & (A >= dens_a_min) & (A <= dens_a_max)
        if mask.size == 0:
            return 0.0, 0.0, 0.0
        s_rel = max(0, min(B.shape[1], split - c0))
        lmask = mask[:, :s_rel]
        rmask = mask[:, s_rel:]
        dl = float(lmask.mean()) if lmask.size else 0.0
        dr = float(rmask.mean()) if rmask.size else 0.0
        dt = float(mask.mean())
        return dl, dr, dt

    ml, mr, mt = _dens_oriented(mid_roi, ang_mid)
    fl, fr, ft = _dens_oriented(far_roi, ang_far)

    mid_dir_norm = float(np.clip(mr - ml, -1.0, 1.0))
    far_dir_norm = float(np.clip(fr - fl, -1.0, 1.0))

    # ----------- Fusion 
    w_near, w_mid, w_far = 0.60, 0.35, 0.05
    raw_dir_score = (w_near * near_dir_norm) + (w_mid * mid_dir_norm) + (w_far * far_dir_norm)
    dir_score = raw_dir_score * float(getattr(hp, "edge_dir_bias", 1.15)) * float(getattr(hp, "dir_sign_gain", 1.0))
    dir_score = float(np.clip(dir_score, -1.0, 1.0))  

    # ----------- Lane confidence 
    near_ratio = float(np.mean(near_roi > 0))
    mid_ratio  = float(np.mean(mid_roi  > 0))
    far_ratio  = float(np.mean(far_roi  > 0))
    lane_conf = float(np.clip((0.5*near_ratio + 0.35*mid_ratio + 0.15*far_ratio)
                              / max(1e-6, float(getattr(hp, "lane_conf_norm", 0.08))), 0.0, 1.0))

    # ----------- Straight suppression 
    try:
        if abs(dir_score) < 0.8:
            r0 = int(h * float(getattr(hp, "flat_roi_row_start", 0.60)))
            fc0 = int(w * float(getattr(hp, "flat_roi_col_lo", 0.30)))
            fc1 = int(w * float(getattr(hp, "flat_roi_col_hi", 0.70)))
            flat_roi = img[r0:, fc0:fc1]
            if flat_roi.size > 0:
                lines, _ = _houghp_adaptive(
                    flat_roi, int(getattr(hp, "hough_threshold", 20)), hp,
                    minLineLength=int(getattr(hp, "flat_min_len", 12)),
                    maxLineGap=int(getattr(hp, "flat_max_gap", 6))
                )
                flat_votes = 0
                if lines is not None:
                    arr = lines.reshape(-1, 4)
                    sdeg = float(getattr(hp, "straight_angle_deg", 18.0))
                    for (x1, y1, x2, y2) in arr:
                        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                        if (ang < sdeg) or (abs(180 - ang) < sdeg):
                            flat_votes += 1
                if flat_votes >= 2:
                    dir_score *= 0.7 # Attenuate score if strong continuous horizontal cues match straight lanes
    except Exception:
        pass

    # --------------------- Confidence / intensity / direction ---------------------
    total_edges = float(2.0 * (near_roi > 0).sum() + 1.0 * (mid_roi > 0).sum() + 0.5 * (far_roi > 0).sum())
    min_edges = int(getattr(hp, "turninfo_total_edges_min", 8))
    edge_factor = float(np.clip(total_edges / max(1, min_edges), 0.0, 1.0))

    dir_conf = float(np.clip(abs(dir_score), 0.0, 1.0))
    if total_edges < min_edges:
        dir_conf *= (0.5 + 0.5 * edge_factor)

    turn_intensity = float(np.clip(abs(dir_score) * float(getattr(hp, "turninfo_imbalance_gain", 1.3)), 0.0, 1.0))

    eps = float(getattr(hp, "direction_deadband", 0.08))
    direction = "right" if dir_score > eps else ("left" if dir_score < -eps else None)
    turn_direction = 1.0 if direction == "right" else (-1.0 if direction == "left" else 0.0)

    return {
        "lane_confidence": float(lane_conf), # How clearly the algorithm sees the lane lines (0.0 = completely blind, 1.0 = perfect sight
        "turn_intensity": float(turn_intensity), # How sharp/intense the upcoming turn is (0.0 = straight road, 1.0 = maximum sharp turn)
        "turn_direction": float(turn_direction), # The direction of the turn (-1.0 = left turn, 1.0 = right turn, 0.0 = straight)
        "direction_confidence": float(dir_conf), # How sure the algorithm is about the turn direction (0.0 = guessing, 1.0 = 100% certain)
        "overall_confidence": float(dir_conf), # Total reliability score combining lane visibility and turn certainty

        # debug
        "near_strength": float(near_strength),
        "mid_density": float(mt),
        "far_density": float(ft),
        "edges_total": int(total_edges),
        "split_px": int(split),
        "direction": direction,
        "lane_conf": float(lane_conf),
        "dir_conf": float(dir_conf),
        "overall_conf": float(dir_conf),
    }

# -------- GOVERNOR 

def _turn_info_from_edges(edge_image: np.ndarray, hp, *,
                          prev_t_ema: float = None,
                          diag_hit: bool = False) -> Tuple[Dict[str, Any], float]:
    """
    Processes turn detection from edge images and applies smoothing to the output.

    It filters out high-frequency noise from raw steering signals using an 
    Exponential Moving Average (EMA) to ensure smooth control updates.
    """
    
    sig = detect_turn_signals(edge_image, hp) # Extract turn features and spatial cues from the edge-detected frame
    t = float(sig.get("turn_intensity", 0.0))

    # Apply Low-Pass Filtering (EMA) to prevent high-frequency control jitter
    if prev_t_ema is not None:
        a = float(getattr(hp, "turn_smooth_alpha", 0.30))
        prev_t_ema = (1.0 - a) * float(prev_t_ema) + a * t
        t = prev_t_ema

    # Construct normalized telemetry dictionary with strict type safety
    info = {
        "lane_confidence":      float(sig.get("lane_confidence", 0.0)),
        "turn_intensity":       float(t),
        "overall_confidence":   float(sig.get("overall_confidence", 0.5)),
        "direction_confidence": float(sig.get("direction_confidence", 0.0)),
        "is_very_flat":         bool(sig.get("is_very_flat", False)),
        "diag_hit":             bool(diag_hit),
    }
    return info, (prev_t_ema if prev_t_ema is not None else t)

def lane_polyfit_features(edge_image: np.ndarray, hp) -> Tuple[float, float, float, float]:
    """
    Fits a second-order polynomial curve to the lane center using edge pixels.
    It scans the bottom half of the image, finds the center point of the lane across 
    multiple rows, and extracts driving telemetry: offset, heading, curvature, and confidence.
    """
    img = edge_image
    # Convert 3-channel images to a single-channel grayscale array
    if img.ndim == 3:
        img = img[0] if img.shape[0] in (1, 3) else img.mean(axis=-1)
        
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Define a Region of Interest (ROI) looking only at the lower 50% of the screen
    r0 = int(h * 0.50)
    roi = img[r0:, :]

    ys, xs = [], []
    n_samples = 18
    # Generate 18 evenly spaced horizontal rows to scan from bottom to top
    rows = np.linspace(roi.shape[0] - 1, 0, n_samples).astype(int)
    
    for ry in rows:
        row = roi[ry, :]
        if row.size == 0:
            continue
        # Use active edge pixels as weights to find the horizontal center of mass
        weights = (row > 0).astype(np.float32) + 1e-6
        col_idx = int(np.round(np.average(np.arange(w), weights=weights)))
        ys.append(ry + r0)
        xs.append(col_idx)

    # We need at least 6 valid data points to safely calculate a polynomial fit
    if len(xs) < 6:
        return 0.0, 0.0, 0.0, 0.0

    y = np.array(ys, dtype=np.float32)
    x = np.array(xs, dtype=np.float32)
    
    # Fit a 2nd-degree polynomial: x = a*y^2 + b*y + c
    a, b, c = np.polyfit(y, x, 2)   

    # Evaluate geometry at a look-ahead target row (70% down the screen)
    y0 = float(h * 0.70)
    dx_dy  = 2 * a * y0 + b  # First derivative (slope)
    d2x_dy = 2 * a           # Second derivative (rate of rotation change)

    # Calculate normalized telemetry metrics scaled roughly between -1.0 and 1.0
    offset    = ((a * y0 * y0 + b * y0 + c) - (w * 0.5)) / (w * 0.5)
    heading   = float(np.tanh(dx_dy / max(1e-6, w * 0.5)))
    curvature = float(np.clip(d2x_dy * (h / max(1.0, w)), -1.0, 1.0))

    # Calculate R-squared (R2) score to verify how well the curve matches the tracked points
    xv = a * y * y + b * y + c
    ss_res = float(np.sum((x - xv)**2))
    ss_tot = float(np.sum((x - np.mean(x))**2)) + 1e-6
    r2   = 1.0 - (ss_res / ss_tot)
    
    # Calculate overall tracking confidence based on R2 fitness and edge pixel density
    dens = float(np.mean(roi > 0))
    conf = float(np.clip(0.6 * r2 + 0.4 * np.clip(dens / 0.08, 0.0, 1.0), 0.0, 1.0))
    
    return float(offset), float(heading), float(curvature), float(conf)


def calculate_progressive_throttle_v2(raw_throttle: float, steering: float,
                                      turn_info: Dict[str, Any],
                                      velocity_history: list, steering_history: list,
                                      max_allowed: float, last_throttle: float,
                                      consecutive_turns: int, stage_name: str,
                                      hp: HyperParams) -> float:
    """
    Dynamically adjusts and smooths the throttle command based on driving telemetry.
    It scales the input throttle down during sharp turns or high curvature, applies a boost on straight sections
    and uses asymmetric temporal smoothing to maintain traction.
    """
    
    # 1) Apply baseline clamping to the raw throttle input
    thr = float(max(0.0, min(float(raw_throttle), float(max_allowed))))
    a = abs(float(steering))

    # 2) Scale down throttle if steering exceeds predefined bracket zones
    if a > hp.steer_bracket_hi:
        thr *= hp.scale_hi
    elif a > hp.steer_bracket_mid:
        thr *= hp.scale_mid

    # 3) Turn-aware scaling (merge)
    t_int   = float(turn_info.get("turn_intensity", 0.0))
    o_conf  = float(turn_info.get("overall_confidence", 0.5))
    d_conf  = float(turn_info.get("direction_confidence", 0.0))
    is_flat = bool(turn_info.get("is_very_flat", False))
    lane_c  = float(turn_info.get("lane_confidence", 0.0))

    cut_ti  = getattr(hp, "turn_intensity_cut", 0.40)
    cut_oc  = getattr(hp, "overall_conf_turn_cut", 0.35)

    # Apply turn-aware scaling based on vision detection certainty
    if t_int > cut_ti and o_conf > cut_oc:
        conf_factor = 0.6 + 0.4 * o_conf

        # Apply extra deceleration if both intensity and direction certainty are high
        hi_prec_factor = 0.9 if (t_int > 0.7 and d_conf > 0.6) else 1.0
        thr *= conf_factor * hi_prec_factor

    # 3b) Apply emergency fast-path multiplier if diagnostic rules trigger
    if bool(turn_info.get("diag_hit", False)):
        try:
            fast_mult = float(getattr(hp, "fast_path_turn_mult", 0.90))
        except Exception:
            fast_mult = 0.90

        # Clamp fast-path multiplier strictly between 0.5 and 1.0 for safety
        if not (0.5 <= fast_mult <= 1.0):
            fast_mult = min(1.0, max(0.5, fast_mult))
        thr *= fast_mult

    # 3c) Apply non-linear curvature-based slowing factors
    curv_for_thr = float(turn_info.get("lane_curvature", 0.0))
    try:
        if not np.isfinite(curv_for_thr):
            curv_for_thr = 0.0
    except Exception:
        pass

    # Clamp curvature metric strictly between 0.0 (straight) and 1.0 (extreme bend)
    curv_for_thr = max(0.0, min(curv_for_thr, 1.0))
    
    # Trigger deceleration only if a curve is present and lane detection confidence is sufficient
    if curv_for_thr > 0.0 and lane_c > 0.20:
        # Fetch sensitivity scaling factor for curvature-based braking
        k = float(getattr(hp, "curv_thr_slow", 0.80))
        
        # Compute smooth non-linear throttle decay factor (inversely proportional to curvature)
        curv_scale = 1.0 / (1.0 + k * curv_for_thr)
        
        # Enforce a safety floor to prevent the vehicle from overslowing or stalling in tight apexes
        curv_floor = float(getattr(hp, "curv_thr_floor", 0.65))  # Enforces a maximum 35% throttle cut
        
        # Apply the conservative scaling factor to the current throttle command
        thr *= max(curv_floor, curv_scale)

    # 4) Straight boost 
    if is_flat and o_conf > getattr(hp, "straight_boost_conf", 0.6) and lane_c > 0.5:
        thr = min(max_allowed, thr * getattr(hp, "straight_boost_gain", 1.1))

    # 5) Stage min throttle 
    stage_min_map = getattr(hp, "stage_min_throttle", {}) or {}
    min_thr = float(stage_min_map.get(stage_name, stage_min_map.get("Basic", 0.08)))

    if consecutive_turns >= 2:
        # max 20% extra progressive
        extra = min(0.2, 0.05 * (consecutive_turns - 1))
        thr *= (1.0 - extra)

    # 6) Clamp the target value between limits before temporal filtering
    target = float(max(min_thr, min(thr, max_allowed)))

    # 7) Apply asymmetric temporal smoothing to decouple acceleration and braking rates
    if not np.isfinite(last_throttle):
        last_throttle = target

    max_up = hp.smooth_up_expert if "Expert" in stage_name else hp.smooth_up_default
    max_down = hp.smooth_down
    if target > last_throttle:
        thr_sm = min(target, last_throttle + max_up)
    else:
        thr_sm = max(target, last_throttle - max_down)

    return float(max(min_thr, min(thr_sm, max_allowed)))


# ---------- FITNESS 

@dataclass
class FitnessMetrics:
    distance: float = 0.0
    stability: float = 0.0
    efficiency: float = 0.0
    exploration: float = 0.0
    consistency: float = 0.0
    lane_conf: float = 0.0
    turn_handling: float = 0.0
    speed_consistency: float = 0.0

def _safe_mean(xs: List[float], default: float = 0.0) -> float:
    return float(np.mean(xs)) if xs else default

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_stage_weights(stage_name: str, config: AdaptiveConfig) -> Dict[str, float]:
    return config.stage_weights.get(stage_name, next(iter(config.stage_weights.values())))

def calculate_improved_fitness(
    genome_metrics: List[Tuple[Genome, FitnessMetrics, List[Tuple[float, float]]]],
    curriculum_stage: str,
    generation: int,
    *,
    action_stride: int = 3, 
    stage_max_steps: int = 500,
    config: Optional[AdaptiveConfig] = None,
) -> List[float]:
    """
    Computes the multi-criteria fitness scores for the entire population.

    It normalizes and weighs driving distance, survival time, speed target accuracy, 
    steering smoothness, and track alignment based on current stage priorities.

    Args:
        genome_metrics: List of tuples containing the genome, its metrics, and action history.
        curriculum_stage: The name of the stage.
        generation: The current evolutionary generation counter.
        action_stride: Frame downsampling rate used during simulation recording.
        stage_max_steps: Maximum step limit for the current track configuration.
        config: Configuration instance containing sub-parameters and weights.

    Returns:
        A list of final float fitness scores bounded between 0.0 and 100.0.
    """

    if not genome_metrics:
        return []

    if config is None:
        raise ValueError("calculate_improved_fitness requires config for stage weights and hparams")

    # Load dynamic scaling variables and weights from the configuration profile
    weights = get_stage_weights(curriculum_stage, config)
    hp = config.hparams
    fitnesses: List[float] = []

    dist_scaling = hp.dist_scaling_by_stage.get(curriculum_stage, 7.0)

    for _, metrics, actions in genome_metrics:
        components = {}

        # Distance component: processed via exponential saturation mapping
        raw_distance = max(0.0, float(metrics.distance))
        components['distance'] = _clamp(1.0 - math.exp(-raw_distance / dist_scaling), 0.0, 1.0)

        # Survival component: ratio of steps cleared vs maximum allowed steps
        est_steps = max(1, len(actions) * action_stride)
        components['survival'] = _clamp(est_steps / float(stage_max_steps), 0.0, 1.0)

        # Speed and Throttle component: evaluates target matching and variance stability
        throttle_vals = [a[1] for a in actions if len(a) > 1]
        if throttle_vals and est_steps > 3:
            target = hp.speed_target_by_stage.get(curriculum_stage, 0.2)
            t_mean = _safe_mean(throttle_vals)
            t_std = float(np.std(throttle_vals))

            # Penalize variance from target speed and penalize erratic pedal jitter
            mean_score = 1.0 - min(1.0, abs(t_mean - target) / max(1e-6, target))
            consistency_score = math.exp(-t_std / 0.12)
            speed_s = _clamp(0.6 * mean_score + 0.4 * consistency_score, 0.0, 1.0)
        else:
            speed_s = 0.0
        components['speed'] = speed_s
        components['throttle'] = speed_s

        # Smoothness component: evaluates the average delta change between steering commands
        if len(actions) > 1:
            steer_vals = [a[0] for a in actions]
            diffs = np.abs(np.diff(steer_vals))
            avg_change = float(np.mean(diffs)) if len(diffs) else 0.0
            components['smoothness'] = _clamp(1.0 - (avg_change / hp.smoothness_norm), 0.0, 1.0)
        else:
            components['smoothness'] = 0.0

        # Road alignment components: track lane centering and curve navigation data
        components['lane'] = _clamp(getattr(metrics, "lane_conf", 0.0), 0.0, 1.0)
        components['turns'] = _clamp(getattr(metrics, "turn_handling", 0.0), 0.0, 1.0)

        # Compute weighted linear combination of all positive driving traits
        base = sum(components[k] * weights[k] for k in weights.keys() if k in components)

        # Penalty system: penalize fatal early crashes and overly defensive stagnation
        penalties = 0.0
        if raw_distance < 1.0 and components['survival'] < 0.15:
            penalties += 0.2
        if curriculum_stage not in ["Basic"] and throttle_vals:
            low_thr_ratio = float(np.mean([t < 0.1 for t in throttle_vals]))
            if low_thr_ratio > 0.7 and raw_distance < 3.0:
                penalties += 0.15 # Deduction for idling or refusing to move forward

        final = _clamp(base - penalties, 0.0, 1.0) # Subtract penalties and apply boundaries

        # NOTE: keys must match curriculum stage names in AdaptiveConfig.curriculum_stages.
        # Unknown stage names silently fall back to 0.0 bonus (no crash, but no bonus either).
        stage_bonus = {"Basic": 0.0, "Early-Intermediate": 0.01, "Late-Intermediate": 0.02,
                       "Advanced": 0.025, "Expert": 0.03}.get(curriculum_stage, 0.0)

        fitnesses.append(100.0 * _clamp(final + stage_bonus, 0.0, 1.0))

    return fitnesses

# ----------- GENOME WRITE-BACK 

def _safe_num(x, default=0.0) -> float:
    """
    Converts the input to a finite float, handling invalid values safely.

    If the input is None, NaN, Infinite, or causes an exception during 
    conversion, the specified default value is returned instead.

    Args:
        x: The input value or object to convert into a float.
        default: The fallback float value used if conversion fails.

    Returns:
        A valid finite float value.
    """
    try:
        v = float(x)
        # Check if the float is a special non-finite value (NaN or Infinity)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        # Catch any type conversion or parsing errors and return the fallback
        return float(default)

def write_back_weights_to_genome(controller: Controller, genome: Genome):
    """
    Extracts weights and parameters from a PyTorch controller and saves them into a Genome.

    It flattens multi-dimensional neural network tensors into a standard Python list 
    and synchronizes metadata tracking variables for external file storage.

    Args:
        controller: The active PyTorch neural network model driving the vehicle.
        genome: The target data object representing the individual's genetic code.
    """
    try:
        with torch.no_grad():
            weights: List[float] = []
            for p in controller.model.parameters():
                weights.extend(p.detach().cpu().numpy().ravel().tolist())
            genome.cnn_weights = weights
            if hasattr(controller, "turn_bias_gain"):
                genome.turn_bias_gain = float(controller.turn_bias_gain.detach().cpu().item())
            if hasattr(controller, "log_std"):
                genome.log_std = [float(v) for v in controller.log_std.detach().cpu().tolist()]
            # --- Keep controller_params in sync for saving ---
            try:
                if getattr(genome, "controller_params", None) is None:
                    genome.controller_params = {}
            except Exception:
                genome.controller_params = {}
            if hasattr(genome, "turn_bias_gain"):
                genome.controller_params["turn_bias_gain"] = float(genome.turn_bias_gain)
            if hasattr(genome, "log_std"):
                ls = getattr(genome, "log_std", None)
                if ls is not None:
                    genome.controller_params["log_std"] = [float(ls[0]), float(ls[1])]
    except Exception as e:
        print(f"⚠ Weight write-back error: {e}")

def save_best_genome(genome: Genome, path: str = "best_genome.json", hp: Optional[HyperParams] = None, input_shape: Optional[Tuple[int,int,int]] = None):
    """
    Saves the target genome alongside its controller priors and hyperparameter metadata.

    This snapshot ensures that future playbacks or evaluations match the exact neural 
    network dimensions and vision preprocessing shapes used during the evolutionary run.

    Args:
        genome: Genome instance selected for saving.
        path: Destination file path for the output JSON file.
        hp: Optional HyperParams snapshot, if missing, the function attempts to infer it.
        input_shape: Image tensor dimensions configuration used by the CNN layers.
    """
    try:
        raw_f = float(getattr(genome, "fitness", 0.0) or 0.0)
        safe_f = 0.0 if (np.isnan(raw_f) or np.isinf(raw_f)) else raw_f

        # ensure controller_params is populated with priors
        cp = dict(getattr(genome, "controller_params", {}) or {})
        if "turn_bias_gain" not in cp and hasattr(genome, "turn_bias_gain"):
            try:
                cp["turn_bias_gain"] = float(getattr(genome, "turn_bias_gain"))
            except Exception:
                pass
        if "log_std" not in cp and hasattr(genome, "log_std"):
            ls = getattr(genome, "log_std", None)
            if isinstance(ls, (list, tuple)) and len(ls) >= 2:
                try:
                    cp["log_std"] = [float(ls[0]), float(ls[1])]
                except Exception:
                    pass

        # Build meta.hp and meta.input_shape
        hp_dict = None
        if hp is None:
            # try to use global config.hparams
            try:
                cfg = globals().get("config", None)
                if cfg is not None and getattr(cfg, "hparams", None) is not None:
                    hp = cfg.hparams
            except Exception:
                pass
        if hp is None:
            try:
                hp = HyperParams()
                #hp.use_hough_helper = False
                #hp.use_sym_control  = False
            except Exception:
                hp = None

        if hp is not None:
            try:
                hp_dict = dict(vars(hp))
            except Exception:
                try:
                    from dataclasses import asdict
                    hp_dict = asdict(hp)
                except Exception:
                    hp_dict = None

        if input_shape is None and hp is not None:
            try:
                input_shape = [1, int(getattr(hp, "resize_h")), int(getattr(hp, "resize_w"))]
            except Exception:
                input_shape = None

        payload = {
            "cnn_weights": getattr(genome, "cnn_weights", None),
            "behavior_params": getattr(genome, "behavior_params", None),
            "controller_params": cp,
            # keep top-level for backward compatibility
            "turn_bias_gain": getattr(genome, "turn_bias_gain", None),
            "log_std": getattr(genome, "log_std", None),
            "fitness": safe_f,
            "meta": {
                "cnn_param_count": len(getattr(genome, "cnn_weights", []) or []),
                "hp": hp_dict,
                "input_shape": input_shape,
            }
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        log(f"Saved best genome to: {path}")
    except Exception as e:
        log(f"Could not save_best_genome: {e}")

def load_best_genome(expected_param_count: Optional[int] = None, path: str = "best_genome.json") -> Optional[Genome]:
    """
    Loads and validates a saved genome from a JSON file.

    It verifies weight array availability, enforces neural network parameter count 
    matching rules, and hydrates legacy properties for backward compatibility.

    Args:
        expected_param_count: The exact number of weight parameters the active model architecture requires.
        path: Path to the target JSON file on disk.

    Returns:
        A fully populated Genome instance if valid otherwise None.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)

        # Extract weights
        w = payload.get("cnn_weights")
        if isinstance(w, dict) and "tolist" in w:
            # in case of numpy json dump style
            w = w.get("tolist")
        if not isinstance(w, list) or len(w) == 0:
            log("⚠ best_genome: no 'cnn_weights' in file.")
            return None

        # Validate param count if provided
        if expected_param_count is not None and len(w) != int(expected_param_count):
            log(f"best_genome param_count mismatch: saved={len(w)} vs expected={int(expected_param_count)}; refusing to load.")
            meta = payload.get("meta") or {}
            if isinstance(meta, dict):
                saved_cnt = meta.get("cnn_param_count")
                if saved_cnt is not None and int(saved_cnt) != int(expected_param_count):
                    log(f"ℹ meta.cnn_param_count={saved_cnt}. Ensure HPs/model match training (input shape, resize_h/w, etc.).")
            return None

        # Construct genome
        param_count = int(expected_param_count) if expected_param_count is not None else len(w)
        g = Genome(param_count, REQ_BP)
        g.cnn_weights = [float(x) for x in w]

        # Behavior params
        bp = payload.get("behavior_params") or [0.0, 0.0]
        if isinstance(bp, list) and len(bp) >= 2:
            g.behavior_params = [float(bp[0]), float(bp[1])]
        else:
            g.behavior_params = [0.0, 0.0]

        # Controller priors
        cp = dict(payload.get("controller_params", {}) or {})
        # Backfill from legacy top-level keys if needed
        if "turn_bias_gain" not in cp and "turn_bias_gain" in payload:
            try:
                cp["turn_bias_gain"] = float(payload["turn_bias_gain"])
            except Exception:
                pass
        if "log_std" not in cp and "log_std" in payload:
            try:
                ls = payload.get("log_std") or [-1.2, -1.2]
                cp["log_std"] = [float(ls[0]), float(ls[1])]
            except Exception:
                pass
        g.controller_params = cp

        # Also hydrate top-level attributes so hasattr(...) checks downstream succeed
        if "turn_bias_gain" in cp:
            try:
                g.turn_bias_gain = float(cp["turn_bias_gain"])
            except Exception:
                pass
        ls = cp.get("log_std")
        if isinstance(ls, (list, tuple)) and len(ls) >= 2:
            try:
                g.log_std = [float(ls[0]), float(ls[1])]
            except Exception:
                pass

        g.fitness = float(payload.get("fitness", 0.0) or 0.0)
        return g
    except Exception as e:
        log(f"Could not load_best_genome: {e}")
        return None
 
def lane_state_from_sym_dbg(sconf: float, sdbg: Any, hp: Any, conf_thr: float = 0.35) -> Optional[Dict[str, Any]]:
    """
    Evaluates the vehicle's alignment and containment relative to detected lane boundaries.

    It computes horizontal pixel displacement between the vehicle center and the track 
    centerline, returning explicit flags for road and lane containment status.

    Args:
        sconf: Confidence score of the structural lane tracking algorithm.
        sdbg: Debug telemetry dictionary containing spatial image coordinates.
        hp: Hyperparameters object holding image dimensional settings.
        conf_thr: Minimum confidence threshold required to process the track state.

    Returns:
        A telemetry dictionary with position flags if tracking is valid; otherwise None.
    """
    # Reject input immediately if metadata block is corrupt or missing
    if not isinstance(sdbg, dict):
        return None
    # Ensure the computer vision tracking cycle completed successfully
    if not sdbg.get("ok", False):
        return None
    # Abort if tracking confidence falls below the strict safety threshold
    if float(sconf) < float(conf_thr):
        return None

    # Fetch spatial horizontal measurements from the input dictionary
    w = float(sdbg.get("image_w", hp.resize_w))
    x_mid = float(sdbg.get("x_mid", 0.5 * w))
    width_px = float(sdbg.get("width_px", 0.0))

    # Validate that the tracked lane has a realistic physics-based width
    half_lane_w = 0.5 * width_px
    if half_lane_w <= 2.0:
        return None

    # Calculate absolute pixel displacement from the camera center line
    car_x = 0.5 * w
    center_error_px = abs(x_mid - car_x)

    # Compile containment metrics based on current lateral error margins
    return {
        # Active if vehicle center stays within full outer road boundaries
        "on_road": int(center_error_px <= 1.00 * half_lane_w),
        # Active if vehicle center remains strictly within the internal 50% safety core
        "in_lane": int(center_error_px <= 0.50 * half_lane_w),
        "center_error_px": float(center_error_px),
        "lane_width_px": float(width_px),
        # Fractional offset: 0.0 represents perfect center, 1.0 marks boundary breach
        "offset_frac": float(center_error_px / max(1e-6, half_lane_w)),
    }


def improved_throttle_control_with_slow_penalty(env, genome: Genome, 
                                                config: AdaptiveConfig, 
                                                curriculum,
                                                track_actions: bool = True, 
                                                action_sample_stride: int = 5
                                                ) -> Tuple[FitnessMetrics, List[Tuple[float, float]]]:
    """Evaluates a genome's driving performance within the environment over a single episode.

    Acts as the main evaluation loop for the genetic algorithm. It processes 
    camera frames using computer vision algorithms (Canny, Lane Line Detection), 
    applies a hybrid control scheme, implements defensive fallbacks, and penalizes
    suboptimal behaviors like slow driving or spinning (donuts).
    """

    hp = config.hparams
    # Handle Gym API discrepancies between older and newer versions dynamically
    try:
        reset_out = env.reset()
        reset_episode_calibration(hp)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    except Exception:
        obs = env.reset()
        reset_episode_calibration(hp)

    # Initialize episode-scoped runtime state tracker
    hp._runtime = {"frames": 0}
    
    # Dynamically adapt Canny edge thresholds to the current track's lighting conditions
    try:
        old_aug = getattr(hp, 'aug_prob', 0.0)
        try:
            hp.aug_prob = 0.0
        except Exception:
            pass

        lo_acc, hi_acc, n = 0, 0, 0
        tmp_obs = obs
        for _ in range(hp.calib_frames):
            img = preprocess(tmp_obs, hp)  
            roi = img  
            lo, hi = _auto_canny_from_sample((roi*255).astype(np.uint8), scale=float(getattr(hp,'canny_scale',1.0)))
            lo_acc += lo 
            hi_acc += hi
            n += 1
            try:
                # Nudge the vehicle slightly forward to gather initial frames
                tmp = env.step((0.0, 0.10))
                tmp_obs = tmp[0] if isinstance(tmp, tuple) else tmp
            except Exception:
                pass
        if n > 0:
            hp.canny_low  = int(lo_acc / n)
            hp.canny_high = int(hi_acc / n)
    finally:
        # Restore original augmentation probability safely
        try:
            hp.aug_prob = old_aug
        except Exception:
            pass

    # Provide a brief initial throttle burst to overcome static friction
    START_THR = 0.12
    early_done = False

    for _ in range(3):
        step_init = env.step((0.0, START_THR))
        hp._runtime["frames"] = hp._runtime.get("frames", 0) + 1

        if isinstance(step_init, tuple) and len(step_init) == 5:
            obs, _, terminated, truncated, _ = step_init
            if bool(terminated) or bool(truncated):
                early_done = True
                break
        else:
            obs, _, done_warm, _ = step_init
            if bool(done_warm):
                early_done = True
                break

    # Immediate exit if the agent crashes during the warm-up phase
    if early_done:
        m = FitnessMetrics()
        return m, []

    controller = Controller(genome, input_shape=(1, hp.resize_h, hp.resize_w))
    _sync_lane_hp_from_genome(hp, genome)  


    # Clamp std from hp
    if hasattr(controller, "log_std"):
        with torch.no_grad():
            controller.log_std.data[0] = controller.log_std.data[0].clamp(min=hp.steer_logstd_min, max=hp.steer_logstd_max)
            controller.log_std.data[1] = controller.log_std.data[1].clamp(min=hp.thr_logstd_min,   max=hp.thr_logstd_max)

    if hasattr(controller, "reset_turn_stabilizer"):
        controller.reset_turn_stabilizer()

    stage = curriculum.get_current_stage()

    # Decide per-episode augmentation ONCE (jitter/blur/Canny)
    _choose_episode_aug(hp)

    max_steps     = stage.get("max_steps", 1000)
    max_throttle  = stage.get("max_throttle", 1.0)
    adaptive_thr  = stage.get("adaptive_throttle", True)
    force_thr     = stage.get("force_throttle", None)

    velocity_history, steering_history = [], []
    consecutive_turns = 0
    last_throttle = float(max_throttle) * 0.5

    done = False
    steps = 0
    actions_taken: List[Tuple[float, float]] = []

    distance = 0.0
    idle_steps = 0
    total_throttle = 0.0
    steering_changes = 0.0
    last_steer = 0.0
    turn_penalty = 0.0

    lane_conf_acc = 0.0
    curve_steps = 0
    curve_penalty_acc = 0.0
    curve_bonus_acc = 0.0

    # Aborts the episode if the agent attempts to exploit rewards by spinning in circles
    SPIN_WIN = 15
    SPIN_STEER_MIN = 0.75
    SPIN_PROGRESS_PER_STEP_MAX = 0.04  
    spin_steers = []
    spin_rewards = []

    consecutive_slow_steps = 0
    total_throttle_penalty = 0.0
    offlane_penalty_acc = 0.0

    turn_int_ema = 0.0

    while not done and steps < max_steps:
        image = preprocess(obs, hp)

        with torch.no_grad():
            steer_raw, thr = controller.predict(image)

        # Perception computed once per step
        _sig = detect_turn_signals(image, hp)
        _sig_lane = _sig

        has_left, has_right = detect_side_lines(image, hp)

        lane_conf_acc += _clamp( float(_sig_lane.get("lane_confidence", 0.0)), 0.0, 1.0)

        _fast_path_hit = detect_diagonal_angle(image, hp) is not None

        turn_info, turn_int_ema = _turn_info_from_edges(image, hp, prev_t_ema=turn_int_ema, diag_hit=_fast_path_hit)

        # Smooth steering via an adaptive Exponential Moving Average (EMA) filter
        alpha_keep = float(getattr(hp, 'steer_ema_keep', 0.50))
        steer = alpha_keep * last_steer + (1.0 - alpha_keep) * float(steer_raw)

        # --- Lane polyfit correction (hybrid) ---
        off, head, curv, conf = lane_polyfit_features(image, hp)

        # Measure current lane width from the symmetric-midline detector (if enabled).
        if bool(getattr(hp, "use_dynamic_lane_width", False)):
            try:
                _w_sconf, _w_soff, _w_shead, _w_sdbg = detect_symmetric_midline(image, hp)
                if isinstance(_w_sdbg, dict) and _w_sdbg.get("ok", False):
                    hp.est_lane_w_px = float(_w_sdbg["width_px"])
            except Exception:
                pass

        # 1. Reliability Check: Ensure the lane detection meets the minimum confidence threshold
        if conf > float(getattr(hp, "lane_conf_thr", 0.50)):

            # 2. Dynamic Track Width Scaling:
            # REF_W: The reference/ideal lane width in pixels.
            # CUR_W: The current estimated lane width in pixels extracted from the frame.
            ref_w_cfg = getattr(hp, 'ref_lane_w_px', None)
            REF_W = float(ref_w_cfg) if ref_w_cfg is not None else (0.60 * hp.resize_w)
            CUR_W = float(getattr(hp, 'est_lane_w_px', REF_W))
            # Compute width scaling factor.
            # As the lane narrows (lower CUR_W), scale_w increases (> 1.0),
            # forcing the controller to respond more aggressively to spatial errors.
            # max(8.0, ...) protects against a critical division-by-zero error.
            scale_w = float(REF_W / max(8.0, CUR_W)) # Value > 1 implies a wider perceived lane

            # 3. Dynamic PD Gains Scaling:
            # Scale baseline hyperparameters dynamically by the computed scale_w factor.
            # Non-linear exponents (0.5 and 1.2) are applied to differentially weight 
            # error components (e.g., placing heavier emphasis on curvature in tight corridors).
            kp_off  = float(hp.lane_kp_off)  * scale_w
            kp_head = float(hp.lane_kp_head) * (scale_w ** 0.5)
            kp_curv = float(hp.lane_kp_curv) * (scale_w ** 1.2)

            # 4. Controller Actuation Math & Error Integration:
            
            # A) Proportional Term (P-term): Minimizes cross-track/lateral error.
            # Applies corrective torque opposite to the displacement sign.
            steer -= hp.offset_sign * (kp_off  * float(off))

            # B) Derivative Term (D-term): Counteracts high-frequency heading changes.
            # Acts as a dampening factor to prevent oscillatory "snaking" behavior down straights.
            steer -= hp.offset_sign * (kp_head * float(head))

            # C) Feed-Forward Curvature Term: Anticipates upcoming bends.
            # Clamping via np.clip shields the steering actuator from sudden optical artifacts,
            # high-frequency noise, or lane detection glitches that could induce catastrophic oversteering.
            steer += (kp_curv * float(np.clip(curv, -hp.curv_thr_cap, hp.curv_thr_cap)))

        steer = float(max(-1.0, min(1.0, steer)))

        if force_thr is not None:
            final_thr = float(force_thr)
        elif not adaptive_thr:
            final_thr = float(max_throttle)
        else:
            turn_info["lane_confidence"] = max(float(turn_info.get("lane_confidence", 0.0)), float(conf))
            turn_info["lane_curvature"]  = max(0.0, min(float(abs(curv)), 1.0)) if conf > hp.lane_conf_thr else 0.0

            final_thr = calculate_progressive_throttle_v2(
                raw_throttle=float(thr),
                steering=float(steer),
                turn_info=turn_info,
                velocity_history=velocity_history,
                steering_history=steering_history,
                max_allowed=max_throttle,
                last_throttle=last_throttle,
                consecutive_turns=consecutive_turns,
                stage_name=str(stage.get("name", "")),
                hp=hp
            )

            # --- LOW-CONFIDENCE FALLBACK ---
            try:
                lane_conf = float(turn_info.get("lane_confidence", 0.0))
            except Exception:
                lane_conf = 0.0

            if lane_conf < 0.15:
                # Blind zone fallback: Cut throttle safely and damp steering behavior
                final_thr = min(final_thr, float(getattr(hp, 'safe_thr', 0.10)))
                steer *= 0.90
                steer = float(max(-1.0, min(1.0, steer)))

            # Maintain runtime dictionary state tracking
            if not hasattr(hp, "_runtime") or not isinstance(hp._runtime, dict):
                hp._runtime = {}
            hp._runtime["speed"] = float(final_thr)
            hp.last_throttle = float(final_thr)

        is_curve = (
            (turn_int_ema > (0.20 if "Basic" in str(stage.get("name","")) else 0.35))
            and (not _sig_lane.get("is_very_flat", False))
            and (float(_sig_lane.get("lane_confidence", 0.0)) > (0.12 if "Basic" in str(stage.get("name","")) else 0.15))
        )
        if _fast_path_hit:
            is_curve = True

        if is_curve:
            curve_steps += 1
            if final_thr > config.hparams.harsh_thr_thr:
                curve_penalty_acc += config.hparams.curve_turn_penalty_gain * abs(steer) * final_thr

            # Evaluate steering alignment quality compared to the track's physical curve direction
            dir_sign  = float(_sig_lane.get("turn_direction", 0.0))  # -1 = Left, +1 = Right
            dir_conf  = float(_sig_lane.get("direction_confidence", 0.0))
            overall_c = float(_sig_lane.get("overall_confidence", 0.0))
            conf_prod = dir_conf * overall_c
            align = max(0.0, float(steer) * dir_sign) # Positive if steering matches curve direction

            # Deadzone gate to ignore minor micro-corrections
            if abs(steer) < config.hparams.turn_bonus_min_steer:
                align = 0.0

            # Scale bonus linearly based on turn intensity and model confidence
            int_gain = min(1.0, float(turn_int_ema) / max(1e-6, config.hparams.turn_bonus_intensity_ref))
            cf_gain  = min(1.0, conf_prod / max(1e-6, config.hparams.turn_bonus_conf_ref))

            curve_bonus_acc += align * int_gain * cf_gain

        velocity_history.append(final_thr)
        steering_history.append(steer)
        if len(velocity_history) > 10:
            velocity_history.pop(0)
            steering_history.pop(0)

        if abs(steer) > 0.3:
            consecutive_turns += 1
        else:
            consecutive_turns = max(0, consecutive_turns - 1)

        if steps > 0:
            steering_changes += abs(steer - last_steer)

        # Penalize jerky, aggressive steering inputs coupled with heavy throttle
        if abs(steer) > config.hparams.harsh_turn_thr and final_thr > config.hparams.harsh_thr_thr:
            turn_penalty += abs(steer) * final_thr

        # Initiates an early turn sequence if Hough space confirms an upcoming corner
        try:
            if bool(getattr(hp, 'use_hough_helper', True)):
                _sig_h = _sig
                dir_sign = float(_sig_h.get('turn_direction', 0.0))  # -1 left, +1 right
                conf_h   = float(_sig_h.get('direction_confidence', 0.0)) * float(_sig_h.get('overall_confidence', 0.0))
                t_ema    = turn_int_ema
                if conf_h > 0.45 and t_ema > 0.30 and abs(dir_sign) > 0.0:
                    steer += float(getattr(hp, 'hough_kp_dir', 0.50)) * (float(getattr(hp,'steer_sign',1)) * float(getattr(hp,'dir_sign_gain',1.0)) * (dir_sign)) * min(1.0, t_ema)
                    # minimum steer if we're certain
                    m = float(getattr(hp, 'min_turn_steer', 0.25))
                    if abs(steer) < m:
                        steer = (1.0 if (float(getattr(hp,'steer_sign',1))*dir_sign) > 0 else -1.0) * m
                steer = float(max(-1.0, min(1.0, steer)))
        except Exception:
            pass

        # Steering aids from lane cues (dashed & symmetric)
        try:
            _dscore, _doff = detect_dashed_centerline(image, hp)
            _sconf, _soff, _shead, _ = detect_symmetric_midline(image, hp)

            if bool(getattr(hp, 'use_dash_control', True)) and float(_dscore) > float(getattr(hp, 'dash_score_thr', 0.55)):
                steer -= hp.offset_sign * hp.dash_kp * _doff

            if bool(getattr(hp, 'use_sym_control', True)) and float(_sconf) > float(getattr(hp, 'sym_conf_thr', 0.6)):
                steer -= hp.offset_sign * hp.sym_kp_off  * _soff
                steer -= hp.offset_sign * hp.sym_kp_head * _shead

            steer = float(max(-1.0, min(1.0, steer)))
        except Exception:
            pass

        last_steer = steer
        last_throttle = final_thr

        action = [float(np.clip(steer, -1.0, 1.0)), float(np.clip(final_thr, 0.0, 1.0))]

        # Accumulate penalties if the vehicle is moving fast without track boundary visual cues
        missing = (not has_left) or (not has_right)  
        moving  = (final_thr > hp.offlane_thr_cut)
        if missing and moving:
            offlane_penalty_acc += hp.offlane_penalty_step * float(final_thr)

        # Penalty for being too slow on straight road
        if action[1] < config.hparams.slow_thr_cut and abs(steer) < config.hparams.slow_steer_cut:
            straight_road = (has_left + has_right) >= 1.5 and detect_flat_lines(image, hp)
            if straight_road:
                consecutive_slow_steps += 1
                if consecutive_slow_steps > config.hparams.slow_steps_grace:
                    total_throttle_penalty += config.hparams.slow_penalty_step * min(
                        consecutive_slow_steps - config.hparams.slow_steps_grace,
                        config.hparams.slow_penalty_cap_steps
                    )
        else:
            consecutive_slow_steps = 0

        if track_actions and (steps % action_sample_stride == 0):
            actions_taken.append((action[0], action[1]))

        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = step_out

        # Terminate early with a flat score if the agent gets trapped in infinite loop spins
        try:
            spin_steers.append(abs(steer))
            spin_rewards.append(float(reward))
            if len(spin_steers) > SPIN_WIN:
                spin_steers.pop(0); spin_rewards.pop(0)
            if len(spin_steers) == SPIN_WIN:
                avg_abs_steer = sum(spin_steers)/SPIN_WIN
                avg_reward    = sum(spin_rewards)/SPIN_WIN
                if avg_abs_steer >= SPIN_STEER_MIN and avg_reward <= SPIN_PROGRESS_PER_STEP_MAX:
                    m = FitnessMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
                    setattr(m, "is_donut", True)
                    return m, actions_taken
        except Exception:
            pass

        if SHOW_DEBUG_WINDOWS:
            try:
                env.render()
            except Exception:
                pass

        # Post-action perception verification
        post_img = preprocess(obs, hp)

        # Lane cues for eval (dashed & symmetric)
        try:
            _dscore, _doff = detect_dashed_centerline(post_img, hp)
            _sconf, _soff, _shead, _sdbg = detect_symmetric_midline(image, hp)

        except Exception:
            _dscore, _doff, _sconf = 0.0, 0.0, 0.0

        conf_eff = max(0.7*float(_dscore), 0.8*float(_sconf))
        post_has_left, post_has_right = detect_side_lines(post_img, hp)

        # Adjust environment reward dynamically with vision confidence scalars
        if post_has_left and post_has_right:
            distance += float(reward)
        elif conf_eff > 0.15:
            distance += float(reward) * float(conf_eff)

        total_throttle += action[1]

        # Step-based micro reward (scaled by confidence & throttle)
        try:
            _thr = float(action[1])
        except Exception:
            _thr = 0.0
        distance += float(getattr(hp, 'step_micro_reward', 0.001)) * float(conf_eff) * max(0.0, _thr)

        if reward < config.hparams.idle_reward_cut:
            idle_steps += 1

        steps += 1

    m = FitnessMetrics()
    if steps > 0:
        total_penalty = (turn_penalty / steps) + total_throttle_penalty + offlane_penalty_acc
        m.distance = max(0.0, distance - total_penalty)
        m.stability = 1.0 - (idle_steps / steps)
        m.efficiency = (total_throttle / steps) * m.stability
        sc = steering_changes / steps
        m.exploration = float(min(1.0, sc))
        m.consistency = float(1.0 / (1.0 + sc))
        m.lane_conf = float(lane_conf_acc / steps)
        if curve_steps > 0:
            per_curve_pen   = curve_penalty_acc / curve_steps
            base_turn_score = _clamp(1.0 - (per_curve_pen / 0.6), 0.0, 1.0)
            per_curve_bonus = curve_bonus_acc / curve_steps
            bonus_normed    = _clamp(per_curve_bonus / max(1e-6, hp.turn_bonus_norm), 0.0, 1.0)
            # Linear combination balancing penalty controls and target turn alignment bonuses
            m.turn_handling = _clamp((1.0 - hp.turn_bonus_mix) * base_turn_score +
                                     hp.turn_bonus_mix * bonus_normed, 0.0, 1.0)
        else:
            m.turn_handling = 0.0

        m.speed_consistency = float(m.efficiency)

    return m, actions_taken

def _eval_distance_only(env, genome: Genome, config: AdaptiveConfig, curriculum, score_fn) -> float:
    """
    Executes a fast, tracking-disabled evaluation of a genome to get its distance metric.

    This function acts as a lightweight wrapper around the main scoring routine, 
    suppressing action logging to maximize execution speed during quick validation passes.

    Args:
        env: The driving simulation environment interface.
        genome: The target Genome instance containing neural net weights to evaluate.
        config: Configuration object containing structural hyperparameters.
        curriculum: The active training curriculum stage.
        score_fn: The core evaluation loop function handle.

    Returns:
        The total validated distance score achieved by the vehicle, defaulting to 0.0 on error.
    """
    try:
        # Run evaluation with tracking deactivated to minimize memory and CPU overhead
        m, _ = score_fn(env, genome, config, curriculum, track_actions=False, action_sample_stride=config.hparams.action_stride_rl)
                        
        # Extract distance metric safely, shielding against potential NaN or missing attribute states
        return _safe_num(getattr(m, "distance", 0.0), 0.0)
        
    except Exception:
        return 0.0
    

def _eval_mean_score(env, genome: Genome, config: AdaptiveConfig, curriculum, score_fn, n: int = 3, seed_offset: int = 0) -> float:
    """
    Evaluates a genome over multiple structural simulation runs to compute a stabilized mean fitness.

    It utilizes deterministic seed shifting for environment consistency, extracts fully mixed 
    fitness weights, and applies a trimmed mean to eliminate statistical outliers.

    Args:
        env: The driving simulation environment interface.
        genome: The target Genome instance containing neural net weights to evaluate.
        config: Configuration object containing structural hyperparameters and seeds.
        curriculum: The active training curriculum sequence controller.
        score_fn: The core evaluation loop function handle.
        n: Total number of evaluation episodes to execute per genome.
        seed_offset: Generational variation padding index used to differentiate pseudorandom series.

    Returns:
        A sanitized, outlier-resistant float representing the structural mean fitness score.
    """
    vals = []
    runs = max(1, int(n))
    stage = curriculum.get_current_stage()
    stage_name = str(stage.get("name", "Basic"))
    max_steps = stage.get("max_steps", 1000)
    
    # Execute structural test iterations over multiple isolated episodes
    for k in range(runs):
        try:
            # Enforce repeatable, pseudorandom environments using antithetic seeding 
            if getattr(config, "use_antithetic_eval_seeds", False):
                seed_everything(config.seed + 31 * (k + 1) + 211 * int(seed_offset))
                
            # Execute standard driving loop tracking active actions
            m, acts = score_fn(env, genome, config, curriculum, track_actions=True, action_sample_stride=config.hparams.action_stride_rl)
            
            # Map physical multi-metric vectors into a single unified score
            fs = calculate_improved_fitness([(genome, m, acts)], stage_name, generation=0, action_stride=config.hparams.action_stride_rl, stage_max_steps=max_steps, config=config)
            
            v = 0.0 if not fs else float(fs[0])
            vals.append(v)
        except Exception:
            vals.append(0.0)
            
    if not vals:
        return 0.0
        
    # Sanitize inputs by converting non-finite mathematical artifacts (NaN/Inf) into neutral zeros
    arr = np.array([float(v) if np.isfinite(float(v)) else 0.0 for v in vals], dtype=np.float64)
    
    # Fall back to standard mean if data bounds are too small for sorting trims
    if len(arr) <= 2:
        return float(np.mean(arr))
        
    # Apply a 15% trimmed mean mechanism to eliminate random glitch profiles and lucky spikes
    k = max(1, int(round(0.15 * len(arr))))
    arr.sort()
    arr = arr[k:len(arr)-k] if len(arr) - 2*k > 0 else arr
    
    return float(np.mean(arr)) if len(arr) > 0 else 0.0

def _eval_mean_distance(env, genome: Genome, config: AdaptiveConfig, curriculum, score_fn, n: int = 2, seed_offset: int = 0) -> float:
    """
    Evaluates a genome over multiple runs to extract a stabilized mean distance.

    Args:
        env: The driving simulation environment interface.
        genome: The target Genome instance containing the weights to evaluate.
        config: Configuration object containing structural hyperparameters and seeds.
        curriculum: The active training curriculum sequence controller.
        score_fn: The core evaluation loop function handle.
        n: Total number of evaluation episodes to execute per genome.
        seed_offset: Generational variation padding index used to differentiate pseudorandom series.

    Returns:
        A finite float representing the true mean distance traversed across the evaluated runs.
    """
    vals = []
    runs = max(1, int(n))
    for k in range(runs):
        try:
            # Antithetic seeds for stability if enabled
            if getattr(config, "use_antithetic_eval_seeds", False):
                seed_everything(config.seed + 17 * (k + 1) + 97 * int(seed_offset))
            v = _eval_distance_only(env, genome, config, curriculum, score_fn)
            try:
                v = 0.0 if v is None else float(v)
            except Exception:
                v = 0.0
            vals.append(v)
        except Exception:
            vals.append(0.0)
    if not vals:
        return 0.0
    # Sanitize
    clean = []
    for v in vals:
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                fv = 0.0
            clean.append(fv)
        except Exception:
            clean.append(0.0)
    return float(np.mean(clean)) if clean else 0.0

def _snapshot_genome(g: Genome) -> Dict[str, Any]:
    """
    Serializes core genome configurations and tracking metrics into a basic dictionary format.

    This function isolates the structural genetic components and fitness data 
    from a Genome instance, facilitating safe checkpointing and JSON storage.

    Args:
        g: The target Genome instance to capture.

    Returns:
        A dictionary containing decoupled, deeply copied configurations and genetic attributes.
    """
    return {
        # Cast the raw CNN weights to a clean list, defaulting to empty if missing
        "cnn_weights": list(getattr(g, "cnn_weights", []) or []),
        # Behavior parameters (lane/turn control gains and perception thresholds)
        "behavior_params": list(getattr(g, "behavior_params", []) or []),
        # Generate a deeply decoupled copy of the controller configurations to avoid pointer mutations
        "controller_params": copy.deepcopy(getattr(g, "controller_params", {}) or {}),
        # Enforce a strict float representation of the final evaluated fitness score
        "fitness": float(getattr(g, "fitness", 0.0) or 0.0),
    }


def _restore_genome_from_snapshot(g: Genome, snap: Dict[str, Any]) -> None:
    """
    Reconstructs a Genome instance's genetic and performance attributes from a serialized snapshot.

    This function performs the inverse operation of snapshotting, parsing raw dictionary data 
    and directly mapping it back onto the target Genome properties for runtime use.

    Args:
        g: The destination Genome instance.
        snap: The source dictionary block containing valid serialized data attributes.
    """
    g.cnn_weights = list(snap.get("cnn_weights", []) or [])
    g.behavior_params = list(snap.get("behavior_params", []) or [])
    g.controller_params = dict(snap.get("controller_params", {}) or {})
    g.fitness = float(snap.get("fitness", 0.0) or 0.0)


# ----------- SELECTION 

def tournament_select(elites: List[Genome], tournament_size: int = 3) -> Genome:
    """
    Selects a single parent Genome from a population using the tournament selection method.

    This routine extracts a random subset of individuals and identifies the winner 
    based on their fitness scores, maintaining genetic diversity while enforcing selection pressure.

    Args:
        elites: A list of active candidate Genome instances available for selection.
        tournament_size: The target number of individuals to sample for the match.

    Returns:
        The winning Genome instance with the highest validated fitness profile.
    """
    if not elites:
        raise ValueError('tournament_select: empty elite list')
    k = max(1, min(int(tournament_size), len(elites)))
    tournament = random.sample(elites, k)
    return max(tournament, key=_fit_or_neginf)


def _fit_or_neginf(g):
    """Return finite float fitness; map None/NaN/Inf to -inf."""
    f = getattr(g, 'fitness', None)
    try:
        f = float(f)
    except (TypeError, ValueError):
        return float('-inf')
    return f if math.isfinite(f) else float('-inf')

def _hash_weights(ws: List[float], k: int = 2048) -> int:
    """
    Generates a deterministic integer hash ID from a neural network weight sequence.

    This routine slices the first k weights, serializes them to raw bytes, and applies 
    the BLAKE2b cryptographic hashing algorithm to produce a unique structural identifier.

    Args:
        ws: The input list of floating-point neural network connection weights.
        k: The maximum number of weight elements to include in the signature mix.

    Returns:
        A unique integer representation of the weight configuration hash, or 0 if empty.
    """
    if not ws:
        return 0
    arr = np.asarray(ws[:k], dtype=np.float32).tobytes()
    h = hashlib.blake2b(arr, digest_size=8).hexdigest()
    return int(h, 16)

def _trimmed_mean(vals: List[float], trim_ratio: float) -> float:
    """
    Computes the trimmed mean of a numeric sequence by filtering outliers and invalid values.
    """
    if not vals:
        return 0.0
    
    # Convert to numpy array and filter out invalid values
    arr = np.array(vals, dtype=np.float64)
    valid_mask = np.isfinite(arr)
    valid_vals = arr[valid_mask]
    
    if len(valid_vals) == 0:
        return 0.0
    
    n = len(valid_vals)
    if n <= 2:
        return float(np.mean(valid_vals))
    
    # Calculate trim amount
    k = int(max(0, min(n // 3, round(n * trim_ratio))))
    
    # Sort and trim
    sorted_vals = np.sort(valid_vals)
    if n - 2 * k > 0:
        trimmed = sorted_vals[k:n - k]
    else:
        trimmed = sorted_vals
    
    return float(np.mean(trimmed)) if len(trimmed) > 0 else 0.0  

def _ema(prev: float, new: float, alpha: float) -> float:
    """
    Computes the Exponential Moving Average (EMA) to smooth sequential updates by blending
    previous states with the incoming signal.

    Args:
        prev: The previously calculated smoothed value.
        new: The raw incoming float value from the current simulation step.
        alpha: The smoothing factor coefficient.

    Returns:
        The newly smoothed float value.
    """
    return (1.0 - alpha) * prev + alpha * new

def _sanitize_population_fitness(pop):
    """Ensure every genome has a numeric fitness for sorting/logging."""
    for g in pop:
        f = getattr(g, "fitness", None)
        try:
            if f is None:
                g.set_fitness(-float("inf"))
            else:
                ff = float(f)
                if ff != ff:  # NaN
                    g.set_fitness(-float("inf"))
                else:
                    g.set_fitness(ff)
        except Exception:
            try:
                g.set_fitness(-float("inf"))
            except Exception:
                pass

# ---------------- EVOLVE 

REQ_BP = 8  # [kp_off, kp_head, kp_curv, curv_thr_slow, lane_conf_thr, canny_scale, hough_thr, diag_roi_row_start]

# NOTE: the bounds used below (and in _sync_lane_hp_from_genome) duplicate
# genome.BP_BOUNDS and Controller._load_behavior_params_from_genome's bounds.
# All three copies must be changed together, or different parts of the
# pipeline will silently disagree on valid behavior-parameter ranges.

def _ensure_bp(g: Genome):
    """
    Sanitizes and validates the behavioral parameter array of a genome.

    Args:
        g: The target Genome instance whose behavior configurations require validation.
    """
    bp = list(getattr(g, "behavior_params", []))
    defaults = [0.30, 0.10, 0.08, 0.80, 0.55, 1.00, 18.0, 0.60]
    # sanitize existing
    clean = []
    for i in range(min(len(bp), REQ_BP)):
        x = float(bp[i])
        if not (x == x and abs(x) < 1e6):  # NaN/Inf guard
            x = defaults[i]
        clean.append(x)
    # pad to REQ_BP
    while len(clean) < REQ_BP:
        clean.append(defaults[len(clean)])
    g.behavior_params = clean

def _sync_lane_hp_from_genome(hp, genome):
    """
    Synchronizes and maps a genome's evolved behavior parameters.
    """
    _ensure_bp(genome)
    bp = genome.behavior_params
    def clamp(x, lo, hi): return float(max(lo, min(hi, x)))
    # 0..4: lane-fit & governor
    hp.lane_kp_off   = clamp(bp[0], 0.00, 0.80)
    hp.lane_kp_head  = clamp(bp[1], 0.00, 0.40)
    hp.lane_kp_curv  = clamp(bp[2], 0.00, 0.50)
    hp.curv_thr_slow = clamp(bp[3], 0.30, 1.50)
    hp.lane_conf_thr = clamp(bp[4], 0.20, 0.80)
    # 5..7: evolvable detection thresholds
    hp.canny_scale        = clamp(bp[5], 0.70, 1.30)
    hp.hough_threshold    = int(round(clamp(bp[6], 10.0, 30.0)))
    hp.diag_roi_row_start = clamp(bp[7], 0.45, 0.75)

def print_best_genome_stats(g, prefix="[SAVE]"):
    w = np.asarray(getattr(g, "cnn_weights", []) or [], dtype=np.float32)
    bp = list(getattr(g, "behavior_params", []) or [])
    cp = getattr(g, "controller_params", {}) or {}

    print(f"{prefix} best genome stats:")
    print(f"  fitness: {getattr(g, 'fitness', None)}")

    if w.size:
        abs_w = np.abs(w)
        print(f"  weights count: {w.size}")
        print(f"  weights mean: {float(np.mean(w)):.6f}")
        print(f"  weights std: {float(np.std(w)):.6f}")
        print(f"  weights range: [{float(np.min(w)):.6f}, {float(np.max(w)):.6f}]")
        print(f"  weights abs mean: {float(np.mean(abs_w)):.6f}")
        print(f"  weights p95_abs: {float(np.quantile(abs_w, 0.95)):.6f}")
        print(f"  weights p99_abs: {float(np.quantile(abs_w, 0.99)):.6f}")
        print(f"  weights zero_frac: {float(np.mean(w == 0.0)):.4f}")
        print(f"  weights nan_count: {int(np.isnan(w).sum())}")
        print(f"  weights inf_count: {int(np.isinf(w).sum())}")
    else:
        print("  weights: EMPTY")

    print(f"  behavior_params: {bp}")

    if len(bp) >= 8:
        print(f"    kp_off:             {bp[0]:.6f}")
        print(f"    kp_head:            {bp[1]:.6f}")
        print(f"    kp_curv:            {bp[2]:.6f}")
        print(f"    curv_thr_slow:      {bp[3]:.6f}")
        print(f"    lane_conf_thr:      {bp[4]:.6f}")
        print(f"    canny_scale:        {bp[5]:.6f}")
        print(f"    hough_threshold:    {bp[6]:.6f}")
        print(f"    diag_roi_row_start: {bp[7]:.6f}")

    print(f"  controller_params: {cp}")
    print(f"  cnn_init_mode: {getattr(g, 'cnn_init_mode', 'N/A')}")
    print(f"  mutate_gaussian: {getattr(g, 'mutate_gaussian', 'N/A')}")
    print(f"  scale_by_std: {getattr(g, 'scale_by_std', 'N/A')}")
    print(f"  weight_clip_k: {getattr(g, 'weight_clip_k', 'N/A')}")
    print(f"  forced_mutation_std_mode: {getattr(g, 'forced_mutation_std_mode', 'N/A')}")

    print(f"  has imitation_status: {hasattr(g, 'imitation_status')}")
    print(f"  imitation_status: {getattr(g, 'imitation_status', 'N/A')}")
    print(f"  last_imitation_loss: {getattr(g, 'last_imitation_loss', 'N/A')}")
    print(f"  imitation_samples: {getattr(g, 'imitation_samples', 'N/A')}")

def evolve(env, config: Optional[AdaptiveConfig] = None, max_generations: int = 20, save_best_path: str = "best_genome.json"):
    """Runs the evolutionary optimization loop.
    
    This function initialized a population of Genomes, evaluates them on the provided environment, 
    applies optional Reinforcement Learning fine-tuning to elites, manages curriculum progression, 
    and handles reproduction via crossover and adaptive mutation.

    Args:
        env: The simulation.
        config: Configuration containing hyperparameters, seeds, and safety guards.
            If None, a default AdaptiveConfig instance is initialized.
        max_generations: Maximum number of evolutionary cycles to execute.
        save_best_path: File path where the best genome will be stored.

    Returns:
        A tuple containing:
            - The best performing Genome found during evolution (or None if failed).
            - A list of dictionaries containing performance metrics tracked per generation.
    """
    if config is None:
        config = AdaptiveConfig()

    hp = config.hparams

    seed_everything(config.seed)
    print("🚀 Starting FAST GA Evolution...")

    curriculum = CurriculumManager(config)

    # Initialize a dummy controller to dynamically extract the required CNN parameter size
    dummy = Controller(input_shape=(1, hp.resize_h, hp.resize_w), load_weights=False)
    cnn_param_count = sum(p.numel() for p in dummy.model.parameters())

    def create_guided_genome():
        """Helper to get a genome warm-started from the saved checkpoint."""
        g = Genome(cnn_param_count, REQ_BP)
        best = load_best_genome(expected_param_count=cnn_param_count, path=save_best_path)
        if best and getattr(best, "cnn_weights", None):
            g.cnn_weights = list(best.cnn_weights)
            if hasattr(best, "turn_bias_gain"):
                g.turn_bias_gain = best.turn_bias_gain
            if hasattr(best, "log_std"):
                g.log_std = best.log_std
            # Inherit behavioral parameters and inject light gaussian jitter for exploration
            bp = getattr(best, "behavior_params", None)
            g.behavior_params = [float(x) for x in (bp if bp is not None else [0.0, 0.0])]
            g.behavior_params = [b + np.random.normal(0, 0.05) for b in g.behavior_params]
            _ensure_bp(g)  
            log("Successfully loaded weights from best_genome!")
        else:
            g.behavior_params = [0.0, 0.0]
            _ensure_bp(g)
        return g

    # --- Population Initialization ---
    population: List[Genome] = []
    guided_count = min(5, config.population_size // 4)

    # Warm-start a part of the population, mutate them progressively to ensure initial variance
    for i in range(guided_count):
        g = create_guided_genome()
        _ensure_bp(g)
        g.mutate(0.10 + 0.02*i, 0.10)
        population.append(g)

    # Fill the remainder of the population with completely random genomes
    while len(population) < config.population_size:
        g = Genome(cnn_param_count, REQ_BP)
        _ensure_bp(g)                       
        population.append(g)

    global_best: Optional[Genome] = None
    global_best_fitness = -float("inf")
    generation_stats: List[Dict] = []
    stagnation_counter = 0

    score_fn = improved_throttle_control_with_slow_penalty
    ACTION_STRIDE = hp.action_stride_ga

    # Hall Of Fame
    hof_hashes: List[int] = []
    last_intervention_gen = -10

    for gen in range(max_generations):
        # Early-gen preset: disable input augmentation for first 3 gens
        if not hasattr(hp, '_aug_prob_base'):
            try:
                hp._aug_prob_base = float(getattr(hp, 'aug_prob', 0.0))
            except Exception:
                hp._aug_prob_base = 0.0
     #   if gen < 5:
     #       try:
     #           hp.aug_prob = 0.0
     #       except Exception:
     #           pass
     #   else:
        try:
            hp.aug_prob = float(hp._aug_prob_base)
        except Exception:
            pass
        gen_start = time.time()
        is_burst = (gen > 0) and (gen % config.explore_burst_period == 0)
        prev_best_fitness = global_best_fitness

        print(f"\n=== Generation {gen+1}/{max_generations} - Stage: {curriculum.get_current_stage()['name']} ===")
        os.environ["CURRENT_GEN"] = str(gen)

        # Early pre-filter: reseed hopeless genomes in first 2 gens
        if gen < 1:
            try:
                reseed_threshold = 3.0  # minimum mean distance over 2 runs
                for idx, g in enumerate(list(population)):
                    md = _eval_mean_distance(env, g, config, curriculum, score_fn, n=2, seed_offset=idx)
                    if md < reseed_threshold:
                        population[idx] = Genome(cnn_param_count, REQ_BP)
                        _ensure_bp(population[idx])
            except Exception as _e:
                log(f"ℹ early pre-filter skipped: {_e}")

        # ----- Phase 1: Evaluation & Scoring ------
        print("Evaluating population...")
        metrics_list: List[Tuple[Genome, FitnessMetrics, List[Tuple[float, float]]]] = []
        for i, genome in enumerate(population):
            try:
                m, actions = score_fn(env, genome, config, curriculum, track_actions=True, action_sample_stride=ACTION_STRIDE)
                # Donut detected? assign -inf fitness and skip normal metrics
                if getattr(m, "is_donut", False):
                    genome.set_fitness(-float("inf"))
                    if PRINT_GENOME_SCORES:
                        log(f"[Gen {gen+1:02d}] {i+1:02d}/{len(population):02d} fit=-inf (donut detected)")
                    continue

                metrics_list.append((genome, m, actions))
                # --- per-genome immediate log ---
                try:
                    stage_now  = curriculum.get_current_stage()
                    stage_name = stage_now.get("name", "Basic")
                    stage_max_steps = stage_now.get("max_steps", 1000)
                    f_single = calculate_improved_fitness(
                        [(genome, m, actions)],
                        stage_name,
                        gen + 1,
                        action_stride=ACTION_STRIDE,
                        stage_max_steps=stage_max_steps,
                        config=config
                    )[0]
                    est_steps = max(1, len(actions) * ACTION_STRIDE)
                    if len(actions) > 1:
                        steer_vals = [a[0] for a in actions]
                        diffs = np.abs(np.diff(steer_vals))
                        smoothness = float(max(0.0, 1.0 - (np.mean(diffs) / config.hparams.smoothness_norm)))
                    else:
                        smoothness = 0.0
                    metrics_dict = {
                        "distance": float(m.distance),
                        "survival": float(min(1.0, est_steps / float(stage_max_steps))),
                        "speed": float(m.speed_consistency),
                        "throttle": float(m.speed_consistency),
                        "smoothness": smoothness,
                        "lane": float(m.lane_conf),
                        "turns": float(m.turn_handling),
                        "steps": int(est_steps),
                    }
                    _log_genome_score(
                        gen_idx=i,
                        gen_num=gen + 1,
                        stage_name=stage_name,
                        fitness=float(f_single),
                        metrics=metrics_dict,
                        pop_size=len(population)
                    )
                except Exception as _e:
                    log(f"i per-genome log skipped for {i}: {_e}")
                # --- end per-genome log ---

            except Exception as e:
                print(f"Error evaluating genome {i}: {e}")
                bad = FitnessMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
                metrics_list.append((genome, bad, []))

        # ---- Phase 2: Fitness ----
        stage = curriculum.get_current_stage()
        fitnesses = calculate_improved_fitness(
            metrics_list, stage.get("name", "Basic"), gen + 1,
            action_stride=ACTION_STRIDE, stage_max_steps=stage.get("max_steps", 1000), config=config
        )

        for i, (genome, m, _) in enumerate(metrics_list):
            f = float(fitnesses[i])
            genome.set_fitness(f)
            if f > global_best_fitness:
                global_best_fitness = f
                global_best = Genome.from_existing(genome)
                print_best_genome_stats(global_best, prefix="[SAVE]")
                save_best_genome(
                    global_best, save_best_path,
                    hp=config.hparams,
                    input_shape=(1, config.hparams.resize_h, config.hparams.resize_w),
                )
                stagnation_counter = 0

        # ----- Phase 2b: Fitness consistency (re-eval) -----
        _sanitize_population_fitness(population)
        population.sort(key=_fit_or_neginf, reverse=True)
        top_count = max(1, int(config.fitness_rollouts_top_frac * len(population)))
        top_count = min(top_count, len(population))

        for j in range(0):         #not used for this version 
            g = population[j]
            f = _eval_mean_score(env, g, config, curriculum, score_fn,
                         n=int(config.fitness_rollouts_top_n),
                         seed_offset=(gen * 1000 + j))
            g.set_fitness(float(f))

        # Resort after smoothing
        _sanitize_population_fitness(population)
        population.sort(key=_fit_or_neginf, reverse=True)
        _sanitize_population_fitness(population)
        fitnesses_sorted = [g.fitness for g in population]

        if fitnesses_sorted and fitnesses_sorted[0] > global_best_fitness:
            global_best_fitness = float(fitnesses_sorted[0])
            global_best = Genome.from_existing(population[0])
            save_best_genome(
                global_best, save_best_path,
                hp=config.hparams,
                input_shape=(1, config.hparams.resize_h, config.hparams.resize_w),
            )

        # Add to HOF if new best
        if population and (len(hof_hashes) == 0 or population[0].fitness > prev_best_fitness ):
            h = _hash_weights(getattr(population[0], 'cnn_weights', []))
            if h and (h not in hof_hashes):
                hof_hashes.insert(0, h)
                hof_cap = int(getattr(getattr(config, 'elite_protection', None), 'hof_max', 5))
                if len(hof_hashes) > hof_cap:
                    hof_hashes.pop()

        # ----- Phase 2c: Elite RL (validation & rollback) -----
        print("[EGA] ENTER Phase 2c (Elite RL)")
        if getattr(config, "elite_rl_enabled", False):
            try:
                protect_k = int(getattr(getattr(config, "elite_protection", None), "top_k", 0))
            except Exception:
                protect_k = 1
            k = max(1, int(float(getattr(config, "elite_rl_frac", 0.20)) * len(population)))
            k = min(k, len(population))
            elites = population[:k]

            for idx, g in enumerate(elites):
                if idx < max(0, protect_k):
                    continue

                print(f"[EGA] RL target rank={idx} (fitness={g.fitness:.2f})")
                snap = _snapshot_genome(g)

                # baseline: full fitness (trimmed mean in multiple runs)
                base_score = _eval_mean_score(
                    env, g, config, curriculum, score_fn,
                    n=int(getattr(config, "elite_eval_runs", 3)),
                    seed_offset=100 + idx)

                # optional RL finetune
                controller = Controller(genome=g, input_shape=(1, hp.resize_h, hp.resize_w), load_weights=True)
                print(f"[EGA] >>> Calling RL fine-tune for elite idx={idx}, steps={getattr(config, 'elite_rl_steps', 2000)}")
                rl_fn = globals().get("rl_finetune", None)
                if callable(rl_fn):
                    try:
                        rl_fn(controller, env, steps=int(getattr(config, "elite_rl_steps", 2000)))
                    except Exception as e:
                        print(f"ℹ RL fine-tune skipped (error): {e}")
                        continue
                else:
                    print("ℹ elite_rl_enabled but rl_finetune() not found.")
                    continue

                tmp = Genome.from_existing(g)                    
                write_back_weights_to_genome(controller, tmp)    

                cand_score = _eval_mean_score(
                    env, tmp, config, curriculum, score_fn,
                    n=int(getattr(config, "elite_eval_runs", 3)),
                    seed_offset=200 + idx)
                
                # Rollback Guard
                margin = float(getattr(config, "elite_accept_margin", 0.0))
                if cand_score + margin < base_score:
                    print(f"[EGA]   ROLLBACK idx={idx} (NOT improved: base={base_score:.2f} > cand={cand_score:.2f})")
                    _restore_genome_from_snapshot(g, snap)    
                else:
                    print(f"[EGA]   ACCEPT idx={idx} (Improved: base={base_score:.2f} -> cand={cand_score:.2f})")
                    write_back_weights_to_genome(controller, g)   
                    try:
                        g.set_fitness(float(cand_score))          
                    except Exception:
                        g.fitness = float(cand_score)

            # Re-sort after changes
            _sanitize_population_fitness(population)
            population.sort(key=_fit_or_neginf, reverse=True)
            _sanitize_population_fitness(population)
            fitnesses_sorted = [g.fitness for g in population]
            if fitnesses_sorted and fitnesses_sorted[0] > global_best_fitness:
                global_best_fitness = float(fitnesses_sorted[0])
                global_best = Genome.from_existing(population[0])
                save_best_genome(
                    global_best, save_best_path,
                    hp=config.hparams,
                    input_shape=(1, config.hparams.resize_h, config.hparams.resize_w),
                )


        # ----- Phase 3: Diversity/Health and possible intervention -----
        actions_per_genome = [acts for (_, _, acts) in metrics_list]
        diversity = DiversityEvaluator.diversity_calculation(population, actions=actions_per_genome)
        health = evaluate_population_health(population, actions_per_genome)

        do_intervene = (config.intervention_enabled and
                        (gen - last_intervention_gen) >= config.intervention_cooldown_gens and
                        (health.collapse_risk >= config.collapse_risk_gate or health.score <= config.health_score_gate))

        if do_intervene:
            print("Population intervention: reseeding some genomes to restore diversity.")
            last_intervention_gen = gen
            # keep top elites, reseed a fraction of the worst
            keep_n = max(3, len(population) // 5)
            survivors = [Genome.from_existing(g) for g in population[:keep_n]]
            reseed_n = max(3, len(population) // 4)
            newcomers = []
            for _ in range(reseed_n):
                _g = Genome(cnn_param_count, REQ_BP)
                _ensure_bp(_g)                     
                try:
                    _g.set_fitness(0.0)
                except Exception:
                     _g.fitness = -float("inf")
                newcomers.append(_g)

            population = survivors + newcomers
            while len(population) < config.population_size:
                g = Genome(cnn_param_count, REQ_BP)
                _ensure_bp(g)                       
                population.append(g)

        # ----- Phase 4: Statistical Metrics Processing -----
        best_f = fitnesses_sorted[0]
        _vals = [float(f) for f in fitnesses_sorted if np.isfinite(float(f))]
        trim_ratio = min(0.10, getattr(config, "fitness_trim_ratio", 0.10))
        mean_f = _trimmed_mean(_vals, trim_ratio)

        if len(_vals) > 2:
            k = int(round(trim_ratio * len(_vals)))
            _t = sorted(_vals)[k:len(_vals)-k] if (len(_vals) - 2*k) > 0 else _vals
            std_f = float(np.std(_t))
        elif len(_vals) == 1:
            std_f = 0.0
        else:
            std_f = 0.0

        gen_time = time.time() - gen_start
        generation_stats.append({
            "generation": gen + 1,
            "best_fitness": best_f,
            "mean_fitness": mean_f,
            "std_fitness": std_f,
            "diversity": diversity,
            "health": {"collapse_risk": health.collapse_risk, "score": health.score},
            "stage": curriculum.get_current_stage()['name'],
            "time": gen_time,
        })
        print(f"best={best_f:.2f} | mean={mean_f:.2f}±{std_f:.2f} | global={global_best_fitness:.2f} "
              f"| div(g)={diversity['genetic']:.3e} fit={diversity['fitness']:.3f} | health={health.score:.2f}/{health.collapse_risk:.2f} | ⏱ {gen_time:.1f}s")

        # ----- Phase 5: Curriculum -----
        if curriculum.should_advance(fitnesses_sorted):
            curriculum.advance_stage()
            stagnation_counter = 0

        # ----- Phase 6: Reproduction & Adaptive Mutation -----
        new_population: List[Genome] = []

        # Define elites and protection 
        elite_count = max(1, int(getattr(config, 'elite_ratio', 0.35) * len(population)))
        elites = population[:elite_count]
        protected_k = max(0, min(int(getattr(getattr(config, 'elite_protection', None), 'top_k', 0)), len(elites)))
        protected_elites = elites[:protected_k]

        if global_best is not None:
            clone = Genome.from_existing(global_best)
            clone._hof_challenger = True
            new_population.append(clone)

        for pe in protected_elites:
            new_population.append(Genome.from_existing(pe))

        elite_keep_extra = max(0, (elite_count // 4) - len(protected_elites))
        for i in range(protected_k, protected_k + elite_keep_extra):
            if i < len(elites):
                if (global_best is not None and
                    getattr(elites[i], "cnn_weights", None) == getattr(global_best, "cnn_weights", None)):
                    continue
                new_population.append(Genome.from_existing(elites[i]))

        # Safety injection if genetic diversity crashes entirely
        div_now = DiversityEvaluator.diversity_calculation(population)
        if div_now['genetic'] < 5e-4:
            rnd = max(3, len(population) // 6)
            for _ in range(rnd):
                new_population.append(Genome(cnn_param_count, REQ_BP))

        fit_array = np.array([g.fitness for g in elites], dtype=np.float32)
        _fit = fit_array[np.isfinite(fit_array)]
        
        trim_ratio = min(0.10, getattr(config, "fitness_trim_ratio", 0.10))
        fit_mean = _trimmed_mean(_fit.tolist(), trim_ratio)

        if _fit.size > 2:
            k = int(round(trim_ratio * _fit.size))
            _t = np.sort(_fit)[k:_fit.size - k] if (_fit.size - 2*k) > 0 else _fit
            fit_std = float(np.std(_t))
        else:
            fit_std = 0.0

        # Standard GA Reproduction Loop via Tournament Selection & Crossover
        while len(new_population) < config.population_size:
            p1 = tournament_select(elites, tournament_size=3)
            p2 = tournament_select(elites, tournament_size=3)
            child = p1.crossover(p2)
            diversity_factor = 1.0 + (1.0 - health.score) * (2.0 if is_burst else 1.0)

            # Early-gen ramp for mutation parameters
            early = min(max(gen, 0), 4)
            target_rate = 0.04 + (0.06 - 0.04) * (early / 4.0)
            target_str  = 0.06 + (0.08 - 0.06) * (early / 4.0)
            base_mrate = target_rate * diversity_factor * (config.explore_burst_mrate_mult if is_burst else 1.0)
            base_mstr  = target_str  * diversity_factor

            parent_fit = max(p1.fitness, p2.fitness)
            scale = 0.8 if fit_std > 0 and parent_fit >= (fit_mean + 0.5 * fit_std) else 1.0
            mrate = base_mrate * scale
            mstr = base_mstr * scale

            # stagnation boost
            if stagnation_counter > 2:
                _boost = 1.0 + 0.25 * (stagnation_counter - 2)
                mrate *= _boost
                mstr  *= _boost

            child.mutate(mrate, mstr)
            _ensure_bp(child) 
            new_population.append(child)

        population = new_population

        # ------ Phase 7: Early stopping -----
        if gen > 0:
            improvement = fitnesses_sorted[0] - generation_stats[-2]["best_fitness"]
            if improvement < config.min_improvement:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            if stagnation_counter >= config.patience:
                print(f"Early stopping: No improvement for {config.patience} generations")
                break

        # ---------- Phase 8: Gentle EMA decays ----------
        # Apply EMA decays only when not in stagnation
        if stagnation_counter == 0:
            config._ema_mut_rate = _ema(config._ema_mut_rate, config.base_mutation_rate * config.mrate_decay_per_gen, config.ema_alpha)
            config.base_mutation_rate = max(config.mrate_min, config._ema_mut_rate)

    print("\nEvolution completed!")
    print(f"Best fitness achieved: {global_best_fitness:.2f}")
    return global_best, generation_stats
