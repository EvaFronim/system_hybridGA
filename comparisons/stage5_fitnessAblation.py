#!/usr/bin/env python3
"""Run the Stage 5 fitness-objective ablation experiment.

This module trains one genetic-algorithm run for a selected fitness mode and
stores the best evolved hybrid driver. Each individual contains two evolved
parts:

* flattened CNN weights
* behavioral/controller parameters

The experiment should be executed N times per fitness mode
(We used 15 runs x 4 modes for statistical comparison).
The saved best genomes should then be compared with the same neutral
evaluation protocol. Raw training fitness values are not directly comparable
across all modes, especially because f3 uses a different rollout/controller pipeline.
a neutral evaluation protocol must be applied post-hoc to all saved genomes for a fair comparison.
pipeline.

Fitness modes:
    f0: Clean rollout with normalized distance-only fitness.
    f1: Clean rollout with distance and behavior-quality terms.
    f2: Clean rollout with behavior-quality terms plus slow/off-lane penalties.
    f3: Final improved rollout using "improved_throttle_control_with_slow_penalty"
        and the full "evolve_ga" fitness calculation.
"""
import os
import json
import math
import argparse
import types
from typing import Dict, List, Tuple, Optional

import numpy as np

from scripts.main_ga import try_make_env

try:
    import torch
except Exception:
    torch = None

from .common_functions import (
    K_REPEATS,
    SEED,
    RUN_ID,
    EVAL_BUDGET,
    POP,
    MAX_STEPS,
    PRIMARY_ENV_ID,
    FALLBACK_ENV_IDS,
    ENV_CONF,
    set_all_seeds,
    make_logger,
    rand_bp_vec,
    mutate_bp_vec,
    crossover_blend_bp_vec,
    bp_vec_to_dict,
    _load_best_genome_from_files,
    HybridIndividual,
    mutate_cnn_vec,
    crossover_uniform_cnn,
    trimmed_mean,
)

try:
    from src.evolve_ga import (
        improved_throttle_control_with_slow_penalty,
        FitnessMetrics,
        AdaptiveConfig,
        calculate_improved_fitness,
        Controller,
        preprocess,
        reset_episode_calibration,
        detect_side_lines,
        detect_flat_lines,
        detect_turn_signals,
        lane_polyfit_features,
        _clamp,
    )

except ImportError:
    print("[ERROR] Cannot import Stage 5 rollout/evaluation dependencies from src.evolve_ga")
    print("[ERROR] Make sure the module is in PYTHONPATH")
    raise

# ------------ Config

# Logical experiment name written in the CSV logs.
SPACE_NAME = "fitness_answer_ablation"

# Default CSV path. main() overwrites this with a mode/stage/run-specific path.
CSV_PATH = os.path.join("logs", "runs_ga_stage5_answerFitness", "trials.csv")


def save_hybrid_genome(path: str,
                       ind: "HybridIndividual",
                       base_controller_params: Dict,
                       meta: Optional[Dict] = None):
    """Save a hybrid GA individual as a JSON genome file.

    Args:
        path: Output JSON path.
        ind: Individual containing cnn_vec and bp_vec.
        base_controller_params: Static controller parameters loaded from the
            pretrained genome.
        meta: Optional experiment metadata to store with the genome.

    The function only serializes data. It does not evaluate or modify the
    individual.
    """
    payload = {
        "cnn_weights": ind.cnn_vec.tolist(),
        "behavior_params": bp_vec_to_dict(ind.bp_vec),
        "controller_params": base_controller_params or {},
        "meta": meta or {},
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)



class MockCurriculum:
    """
    Tiny adapter that mimics the curriculum object expected by evolve_ga.

    Stage 5 does not run a real curriculum scheduler. It only needs to provide
    the current stage settings so the existing controller/evaluation function
    can be reused without rewriting it.
    """
    def __init__(self, stage_name: str = "Basic"):
        self.stage_name = stage_name
    
    def get_current_stage(self):
        return {
            "name": self.stage_name,
            "max_steps": MAX_STEPS,
            "max_throttle": 0.10,
            "adaptive_throttle": True,
            "force_throttle": None,
        }

class MockConfig(AdaptiveConfig):
    """
    AdaptiveConfig wrapper whose hyperparameters come from one genome.

    The rollout function already knows how to use AdaptiveConfig. Instead of
    writing a new controller path for Stage 5, this class starts from the normal
    evolve_ga configuration and overwrites any matching hparam with the genome's
    behavioral parameter values.
    """
    def __init__(self, bp_dict: Dict[str, float]):
        super().__init__()
        hp = self.hparams
        
        for k, v in bp_dict.items():
            if hasattr(hp, k):
                setattr(hp, k, v)


def _safe_env_step(env, action):
    """
    Step helper compatible with both old and new Gym APIs.

    Returns:
        obs, reward, done, info
    """
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated) or bool(truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def _safe_env_reset(env, hp):
    """
    Reset helper compatible with both old and new Gym APIs.
    Also resets episode-scoped calibration state when available.
    """
    try:
        reset_out = env.reset()
        try:
            reset_episode_calibration(hp)
        except Exception:
            pass
        return reset_out[0] if isinstance(reset_out, tuple) else reset_out
    except Exception:
        obs = env.reset()
        try:
            reset_episode_calibration(hp)
        except Exception:
            pass
        return obs


def clean_policy_rollout(env,
                          genome,
                          config: AdaptiveConfig,
                          curriculum,
                          track_actions: bool = True,
                          action_sample_stride: int = 5) -> Tuple[FitnessMetrics, List[Tuple[float, float]]]:
    """
    Clean rollout used by F0-F2.

    This intentionally avoids the full rule-assisted controller used by the
    improved system. The vehicle action is produced mainly by the evolved
    Controller. The function keeps only action clipping and metric collection.

    Important design rule:
        Do not subtract slow/off-lane/turn penalties from distance here.
        Those penalties belong in compute_fitness(), otherwise F0 is no longer
        a true distance-only objective.

    What is deliberately NOT applied here:
        - lane-polyfit steering correction
        - Hough turn helper
        - dashed/symmetric centerline steering corrections
        - low-confidence steering/throttle fallback
        - progressive throttle controller
        - in-rollout slow-driving penalty
        - in-rollout off-lane penalty
        - turn-alignment bonus

    Fitness shaping for F0-F2 is applied later in compute_fitness(...), not
    inside the rollout.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for Controller-based rollout.")

    hp = config.hparams
    obs = _safe_env_reset(env, hp)

    # Runtime state is kept because some preprocessing/controller utilities may
    # expect it, but no full assisted-control state machine is used here.
    hp._runtime = {"frames": 0}

    stage = curriculum.get_current_stage()
    max_steps = int(stage.get("max_steps", MAX_STEPS))
    max_throttle = float(stage.get("max_throttle", 1.0))
    force_thr = stage.get("force_throttle", None)

    controller = Controller(genome, input_shape=(1, hp.resize_h, hp.resize_w))

    if hasattr(controller, "log_std"):
        with torch.no_grad():
            controller.log_std.data[0] = controller.log_std.data[0].clamp(
                min=hp.steer_logstd_min,
                max=hp.steer_logstd_max,
            )
            controller.log_std.data[1] = controller.log_std.data[1].clamp(
                min=hp.thr_logstd_min,
                max=hp.thr_logstd_max,
            )

    if hasattr(controller, "reset_turn_stabilizer"):
        controller.reset_turn_stabilizer()

    done = False
    steps = 0
    actions_taken: List[Tuple[float, float]] = []

    distance = 0.0
    idle_steps = 0
    total_throttle = 0.0
    steering_changes = 0.0
    last_steer = 0.0

    lane_conf_acc = 0.0
    turn_handling_acc = 0.0
    turn_eval_steps = 0
    offlane_steps = 0
    slow_steps = 0

    while not done and steps < max_steps:
        image = preprocess(obs, hp)

        with torch.no_grad():
            steer_raw, thr_raw = controller.predict(image)

        # Basic action mapping only. This is intentionally much simpler than
        # calculate_progressive_throttle_v2 and the full assisted controller.
        steer = float(np.clip(float(steer_raw), -1.0, 1.0))

        if force_thr is not None:
            final_thr = float(force_thr)
        else:
            # Controller throttle may be stochastic / unconstrained depending
            # on implementation. Clip it into the legal simulator range and
            # respect the curriculum max throttle.
            final_thr = float(np.clip(float(thr_raw), 0.0, max_throttle))

        action = [
            float(np.clip(steer, -1.0, 1.0)),
            float(np.clip(final_thr, 0.0, 1.0)),
        ]

        if track_actions and (steps % action_sample_stride == 0):
            actions_taken.append((action[0], action[1]))

        obs, reward, done, _info = _safe_env_step(env, action)

        # Basic perception is used only to measure quality, not to correct the
        # action. That keeps the rollout simple while still giving F1/F2/F3
        # comparable metrics.
        try:
            post_img = preprocess(obs, hp)
        except Exception:
            post_img = image

        try:
            has_left, has_right = detect_side_lines(post_img, hp)
        except Exception:
            has_left, has_right = False, False

        try:
            sig = detect_turn_signals(post_img, hp)
            lane_conf = _clamp(float(sig.get("lane_confidence", 0.0)), 0.0, 1.0)
        except Exception:
            sig = {}
            lane_conf = 0.0

        lane_conf_acc += lane_conf

        # Raw progress signal. No internal penalty or reward shaping is applied
        # here; fitness shaping is handled only by compute_fitness().
        distance += float(reward)

        # Optional turn-quality measurement only. No steering correction.
        try:
            off, head, curv, poly_conf = lane_polyfit_features(post_img, hp)
            turn_direction = float(sig.get("turn_direction", 0.0))
            direction_conf = float(sig.get("direction_confidence", 0.0))
            if abs(turn_direction) > 0.0 and direction_conf > 0.25:
                align = max(0.0, steer * turn_direction)
                turn_handling_acc += align * direction_conf * max(lane_conf, float(poly_conf))
                turn_eval_steps += 1
        except Exception:
            pass

        total_throttle += action[1]

        if steps > 0:
            steering_changes += abs(steer - last_steer)

        if reward < config.hparams.idle_reward_cut:
            idle_steps += 1

        # These are measured but not directly subtracted here. F2 can penalize
        # slow/off-lane behavior through compute_fitness.
        try:
            if action[1] < config.hparams.slow_thr_cut and abs(steer) < config.hparams.slow_steer_cut:
                if detect_flat_lines(post_img, hp):
                    slow_steps += 1
        except Exception:
            pass

        if (not has_left or not has_right) and action[1] > getattr(hp, "offlane_thr_cut", 0.05):
            offlane_steps += 1

        last_steer = steer
        steps += 1

    m = FitnessMetrics()
    if steps > 0:
        m.distance = max(0.0, float(distance))
        m.stability = float(1.0 - (idle_steps / steps))
        m.efficiency = float((total_throttle / steps) * m.stability)

        sc = steering_changes / steps
        m.exploration = float(min(1.0, sc))
        m.consistency = float(1.0 / (1.0 + sc))
        m.lane_conf = float(lane_conf_acc / steps)
        m.turn_handling = float(_clamp(turn_handling_acc / max(1, turn_eval_steps), 0.0, 1.0))
        m.speed_consistency = float(m.efficiency)

        # Extra diagnostic fields. compute_fitness currently uses lane_conf and
        # efficiency for F2 penalties, but these are useful for logging/debugging
        # if you later extend the logger payload.
        setattr(m, "slow_rate", float(slow_steps / steps))
        setattr(m, "offlane_rate", float(offlane_steps / steps))

    return m, actions_taken


def evaluate_individual_custom(env,
                               ind: HybridIndividual,
                               base_controller_params: Dict,
                               trial_idx: int,
                               fitness_mode: str = "f2",
                               stage_name: str = "Basic") -> float:
    """
    Evaluate one Stage 5 hybrid individual over K repeated rollouts.

    Stage 5 is implemented as an incremental ablation:

        F0-F2:
            Use the same clean rollout and differ only in the scalar fitness
            used for GA selection.

        F3:
            Use the final-stage improved_throttle_control_with_slow_penalty
            rollout and full evolve_ga-style fitness, by design.

    Returns:
        Scalar fitness value for the evaluated individual.
    """
    bp_dict = bp_vec_to_dict(ind.bp_vec)

    curriculum = MockCurriculum(stage_name)
    config = MockConfig(bp_dict)

    fitness_mode = fitness_mode.lower()

    all_metrics: List[FitnessMetrics] = []
    all_actions: List[List[Tuple[float, float]]] = []

    for r in range(K_REPEATS):
        s = SEED + int(RUN_ID) * 100000 + (trial_idx + 1) * 100 + r
        set_all_seeds(s)

        mock_genome = types.SimpleNamespace()
        mock_genome.cnn_weights = ind.cnn_vec.tolist()
        mock_genome.behavior_params = ind.bp_vec.tolist()
        mock_genome.controller_params = base_controller_params

        if fitness_mode == "f3":
            # Full system: rule-assisted perception/control + full fitness.
            metrics, actions = improved_throttle_control_with_slow_penalty(
                env=env,
                genome=mock_genome,
                config=config,
                curriculum=curriculum,
                track_actions=True,
                action_sample_stride=5,
            )
        else:
            # Clean baselines: no internal slow/off-lane/turn penalties.
            # Fitness shaping is applied later by compute_fitness.
            metrics, actions = clean_policy_rollout(
                env=env,
                genome=mock_genome,
                config=config,
                curriculum=curriculum,
                track_actions=True,
                action_sample_stride=5,
            )

        if hasattr(metrics, "is_donut") and getattr(metrics, "is_donut", False):
            metrics.distance = 0.0

        all_metrics.append(metrics)
        all_actions.append(actions)

    custom_fit = compute_fitness(
        metrics_list=all_metrics,
        actions_list=all_actions,
        mode=fitness_mode,
        stage_name=stage_name,
        config=config,
    )

    return float(custom_fit)

# ============================================================
# Fitness Functions (F0 → F3)
# ============================================================

def compute_fitness(
    metrics_list: List[FitnessMetrics],
    actions_list: List[List[Tuple[float, float]]],
    mode: str,
    stage_name: str,
    config: Optional[AdaptiveConfig] = None,
) -> float:
    """
    Compute the Stage 5 ablation fitness from repeated rollout metrics.

    This function computes the scalar selection fitness from rollout metrics.
    In this Stage 5 design, F0-F2 use clean rollout metrics, while F3 uses
    metrics from the final improved rollout. The mode controls how metrics are
    converted into a scalar GA selection score.

    Ablation ladder:
        F0:
            Normalized distance only.

        F1:
            F0 plus behavioral quality components:
            stability, efficiency, consistency, and speed consistency.

        F2:
            F1 plus explicit penalties for slow driving and poor lane
            confidence.

        F3:
            Full evolve_ga-style fitness through calculate_improved_fitness,
            including lane and turn rewards.

    Args:
        metrics_list:
            List of FitnessMetrics objects, one per repeated rollout.
        actions_list:
            List of sampled action sequences, one per repeated rollout.
            Used only by F3, because calculate_improved_fitness expects
            action history.
        mode:
            Ablation mode. Expected values: "f0", "f1", "f2", "f3".
        stage_name:
            Curriculum/stage name used for distance scaling and full fitness.

    Returns:
        Scalar fitness value.
    """
    if not metrics_list:
        return 0.0

    mode = mode.lower()

    # Aggregate metrics across all repeated rollouts.
    distances      = [float(m.distance)        for m in metrics_list]
    stabilities    = [float(m.stability)       for m in metrics_list]
    efficiencies   = [float(m.efficiency)      for m in metrics_list]
    explorations   = [float(m.exploration)     for m in metrics_list]
    consistencies  = [float(m.consistency)     for m in metrics_list]
    lane_confs     = [float(m.lane_conf)       for m in metrics_list]
    turn_handlings = [float(m.turn_handling)   for m in metrics_list]
    speed_cons = [float(getattr(m, "speed_consistency", m.efficiency)) for m in metrics_list]

    mean_dist  = float(np.mean(distances))
    mean_stab  = float(np.mean(stabilities))
    mean_eff   = float(np.mean(efficiencies))
    mean_expl  = float(np.mean(explorations))
    mean_cons  = float(np.mean(consistencies))
    mean_lane  = float(np.mean(lane_confs))
    mean_turn  = float(np.mean(turn_handlings))
    mean_speed = float(np.mean(speed_cons))

    # Distance normalization uses the same stage-specific scaling convention
    # as the adaptive fitness configuration.
    cfg = config if config is not None else AdaptiveConfig()
    dist_scale = cfg.hparams.dist_scaling_by_stage.get(stage_name, 7.0)

    # ------------------------------------------------------------
    # F0: normalized distance only
    # ------------------------------------------------------------
    if mode == "f0":
        distance_norm = 1.0 - math.exp(-mean_dist / dist_scale)
        return 100.0 * float(np.clip(distance_norm, 0.0, 1.0))

    # ------------------------------------------------------------
    # F1: F0 + behavioral components
    # ------------------------------------------------------------
    if mode == "f1":
        distance_norm = 1.0 - math.exp(-mean_dist / dist_scale)

        # Behavioral quality excluding lane and turn information.
        behavioral_score = (
            0.25 * mean_stab +      # steering/action stability
            0.25 * mean_eff +       # throttle efficiency
            0.25 * mean_cons +      # rollout consistency
            0.25 * mean_speed       # speed consistency
        )

        # Keep distance dominant, while still rewarding smoother behavior.
        combined = 0.70 * distance_norm + 0.30 * behavioral_score
        return 100.0 * float(np.clip(combined, 0.0, 1.0))

    # ------------------------------------------------------------
    # F2: F1 + slow-driving and off-lane penalties
    # ------------------------------------------------------------
    if mode == "f2":
        distance_norm = 1.0 - math.exp(-mean_dist / dist_scale)
        behavioral_score = (
            0.25 * mean_stab +
            0.25 * mean_eff +
            0.25 * mean_cons +
            0.25 * mean_speed
        )

        # Penalize genomes that move too slowly despite surviving.
        slow_rates = [float(getattr(m, "slow_rate", 0.0)) for m in metrics_list]
        offlane_rates = [float(getattr(m, "offlane_rate", 0.0)) for m in metrics_list]
        mean_slow_rate = float(np.mean(slow_rates))
        mean_offlane_rate = float(np.mean(offlane_rates))

        slow_penalty = 0.10 * mean_slow_rate

        # Penalize low lane confidence and measured off-lane behavior.
        low_lane_penalty = 0.0
        if mean_lane < 0.7:
            low_lane_penalty = 0.10 * (0.7 - mean_lane)
        offlane_penalty = 0.10 * mean_offlane_rate

        combined = 0.70 * distance_norm + 0.30 * behavioral_score
        combined -= (slow_penalty + low_lane_penalty + offlane_penalty)

        return 100.0 * float(np.clip(combined, 0.0, 1.0))

    # ------------------------------------------------------------
    # F3: full evolve_ga-style fitness
    # ------------------------------------------------------------
    if mode == "f3":
        repeat_scores: List[float] = []

        for m, actions in zip(metrics_list, actions_list):
            genome_metrics = [(None, m, actions)]

            scores = calculate_improved_fitness(
                genome_metrics,
                curriculum_stage=stage_name,
                generation=0,
                action_stride=5,
                stage_max_steps=MAX_STEPS,
                config=cfg,
            )

            if scores:
                score = float(scores[0])
                if np.isfinite(score):
                    repeat_scores.append(score)

        if not repeat_scores:
            return 0.0

        return float(trimmed_mean(repeat_scores, trim_ratio=0.10))

    raise ValueError(f"Unknown fitness mode: {mode}")

def individual_log_payload(
    ind: HybridIndividual,
    fitness_mode: str,
    stage_name: str,
) -> Dict:
    """
    Build a compact JSON-serializable summary for CSV logging.

    Logging the full CNN vector every trial would make the CSV huge. Instead,
    we store the behavior params fully and only basic statistics for the CNN
    vector. The full best genomes are saved separately as JSON files.
    """
    return {
        "fitness_mode": fitness_mode,
        "stage_name": stage_name,
        "bp": bp_vec_to_dict(ind.bp_vec),
        "cnn": {
            "dim": int(ind.cnn_vec.size),
            "mean": float(np.mean(ind.cnn_vec)),
            "std": float(np.std(ind.cnn_vec)),
            "min": float(np.min(ind.cnn_vec)),
            "max": float(np.max(ind.cnn_vec)),
        },
    }


def main():
    """
    Run one Stage 5 GA experiment.

    Execution flow:
        1. Parse CLI args and choose fitness mode/stage.
        2. Load a pretrained genome, mainly to obtain the CNN weight vector.
        3. Initialize a hybrid population (init_mode=pretrained):
             CNN = pretrained weights + small Gaussian noise  (σ=0.01)
             BP  = random behavior-parameter vector
           Or (init_mode=random):
             CNN = random Gaussian weights  (σ=0.05, same shape)
             BP  = random behavior-parameter vector
        4. Evaluate generation 0.
        5. Repeat tournament selection, crossover, fixed mutation, evaluation.
        6. Keep a one-individual Hall-of-Fame global best.
        7. Save logs and the final best genome for offline evaluation.

    Stage 5 deliberately uses fixed mutation, not the adaptive EMA mutation of
    Stage 4b, because here the variable under test is the fitness formulation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_mode",type=str,default="pretrained",choices=["pretrained", "random"],
        help="CNN initialization mode: pretrained or random template-shaped initialization",
    )
    parser.add_argument("--pretrained_npy", type=str, default=None)
    parser.add_argument("--pretrained_json", type=str, required=True)
    parser.add_argument("--sim_host", type=str, default=ENV_CONF["host"])
    parser.add_argument("--sim_port", type=int, default=ENV_CONF["port"])
    parser.add_argument(
        "--fitness_mode",
        type=str,
        default=os.environ.get("FITNESS_MODE", "f2"),
        choices=["f0", "f1", "f2", "f3"],
        help="Mode: f0=clean+distance, f1=clean+behavioral, f2=clean+penalties, f3=final improved system",
    )
    parser.add_argument("--stage_name", type=str, default="Basic")
    args = parser.parse_args()

    run_id_int = int(RUN_ID)
    # RUN_ID shifts the base seed so repeated runs are reproducible but distinct.
    effective_seed = int(SEED) + run_id_int

    set_all_seeds(effective_seed)

    if torch is not None:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # Single RNG object for all GA stochastic operations in this run.
    rng = np.random.default_rng(effective_seed)

    fitness_mode = args.fitness_mode.lower()
    stage_name = args.stage_name

    global CSV_PATH
    CSV_PATH = os.path.join(
        "logs",
        "runs_ga_stage5_answerFitness",
        f"trials_{stage_name}_{fitness_mode}_run{RUN_ID}.csv",
    )

    print(f"[Stage5] Fitness mode: {fitness_mode}")
    print(f"[Stage5] Stage: {stage_name}")
    print(f"[Stage5]  F0-F2 use clean_policy_rollout")
    print(f"[Stage5]  F3 uses improved_throttle_control_with_slow_penalty")
    print(f"[Stage5]  Final comparison must use neutral evaluation of saved best genomes")

    # Env override
    ENV_CONF["host"] = args.sim_host
    ENV_CONF["port"] = int(args.sim_port)

    # ----- Load pretrained genome (CNN weights + controller params) -----
    genome = _load_best_genome_from_files(args.pretrained_npy, args.pretrained_json)
    if genome is None:
        raise RuntimeError("Could not load genome")

    template_weights = np.asarray(genome.cnn_weights, dtype=np.float32).ravel()
    cnn_dim = template_weights.size
    base_controller_params = dict(getattr(genome, "controller_params", {}) or {})

    if args.init_mode == "pretrained":
        cnn_base = template_weights
    elif args.init_mode == "random":
        cnn_base = rng.normal(0.0, 0.05, size=cnn_dim).astype(np.float32)
    else:
        raise ValueError(f"Unknown init_mode: {args.init_mode}")

    print(f"Loaded CNN weights: {cnn_dim} params")

    # ----- Env + logger -----
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)

    try:
        _run_stage5(
            env, logger, rng, args,
            base_controller_params=base_controller_params,
            cnn_base=cnn_base, fitness_mode=fitness_mode,
            stage_name=stage_name, effective_seed=effective_seed,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_stage5(env, logger, rng, args, base_controller_params,
                 cnn_base, fitness_mode, stage_name, effective_seed):
    print("\n" + "=" * 60)
    print("Stage5 Answer Fitness Function Ablation")
    print("=" * 60)
    print(f"Population: {POP}")
    print(f"Eval budget: {EVAL_BUDGET}")
    print(f"K-repeats: {K_REPEATS}")
    print(f"Fitness: {fitness_mode}")
    print(f"F0 = clean rollout + normalized raw distance")
    print(f"F1 = clean rollout + behavioral components")
    print(f"F2 = clean rollout + explicit penalties")
    print(f"F3 = final improved rollout + full evolve_ga fitness")
    print(f"CSV: {CSV_PATH}")
    print("=" * 60 + "\n")

    # ----- Init population -----
    pop = max(1, int(POP))

    # EVAL_BUDGET counts individual evaluations, not generations.
    # With population size POP, this is the maximum number of full generations.
    max_generations = max(1, math.ceil(EVAL_BUDGET / pop))

    population: List[HybridIndividual] = []
    for _ in range(pop):
        # CNN = pretrained weights plus small noise.
        if args.init_mode == "pretrained":
            cnn_init = cnn_base + rng.normal(0.0, 0.01, size=cnn_base.shape)
        else:
            cnn_init = rng.normal(0.0, 0.05, size=cnn_base.shape).astype(np.float32)

        bp_init = rand_bp_vec(rng)
        population.append(HybridIndividual(cnn_init, bp_init))

    fitnesses: List[Optional[float]] = [None] * pop

    trial = 0
    best_so_far = -1e12
    global_best: Optional[HybridIndividual] = None
    global_best_fit = -1e12
    global_best_gen = -1

    # ----- Initial evaluation (GEN 0) -----
    print("[INFO] Initial evaluation (GEN 0)")
    for i in range(pop):
        if trial >= EVAL_BUDGET:
            break

        custom_fit = evaluate_individual_custom(
            env, population[i], base_controller_params, trial,
            fitness_mode, stage_name
        )

        fitnesses[i] = custom_fit
        trial += 1

        params = individual_log_payload(population[i], fitness_mode, stage_name)
        best_so_far = logger(trial, RUN_ID, params, custom_fit)

        print(f"[GEN 0] ind={i+1}/{pop} fit={custom_fit:.2f} best={best_so_far:.2f}")

    if any(f is not None for f in fitnesses):
        best_idx = int(np.nanargmax([f if f is not None else -1e12 for f in fitnesses]))
        global_best = population[best_idx].copy()
        global_best_fit = float(fitnesses[best_idx])
        global_best_gen = 0

        save_hybrid_genome(
            path=os.path.join(
                "saved_policies",
                "stage5_answerFitness",
                f"run{RUN_ID}_gen0_{stage_name}_{fitness_mode}_fit{global_best_fit:.6f}.json",
            ),
            ind=global_best,
            base_controller_params=base_controller_params,
        )

    gen = 1
    while trial < EVAL_BUDGET and gen <= max_generations:
        print(f"\n[INFO] Generation {gen}/{max_generations}")

        def tournament() -> HybridIndividual:
            """
            Select one parent with size-3 tournament selection.

            We sample a few candidates and copy the one with the best known
            fitness. This gives selection pressure without always picking only
            the global best.
            """
            idxs = rng.choice(len(population), size=min(3, len(population)), replace=False)
            best_idx_local = max(
                idxs,
                key=lambda j: fitnesses[j] if fitnesses[j] is not None else -1e12,
            )
            return population[best_idx_local].copy()

        # Hall-of-Fame elite
        elites: List[HybridIndividual] = []
        if global_best is not None:
            elites.append(global_best.copy())

        # offspring via crossover+mutation
        offspring: List[HybridIndividual] = []
        while len(offspring) + len(elites) < pop and trial < EVAL_BUDGET:
            p1 = tournament()
            p2 = tournament()
            child_cnn = crossover_uniform_cnn(p1.cnn_vec, p2.cnn_vec, rng)
            child_bp = crossover_blend_bp_vec(p1.bp_vec, p2.bp_vec, rng, alpha=0.2)

            # Fixed mutation setup inherited from Stage 4a:
            # - CNN gets small sparse Gaussian perturbations.
            # - behavior params mutate more aggressively because they are low-dimensional.
            child_cnn = mutate_cnn_vec(child_cnn, rng, sigma=0.01, p=0.05)
            child_bp = mutate_bp_vec(child_bp, rng, sigma_scale=0.10, p=0.25)

            offspring.append(HybridIndividual(child_cnn, child_bp))

        new_population: List[HybridIndividual] = elites + offspring[: max(0, pop - len(elites))]
        new_fitnesses: List[Optional[float]] = [None] * len(new_population)

        # Evaluate new population
        for i in range(len(new_population)):
            if trial >= EVAL_BUDGET:
                break

            custom_fit = evaluate_individual_custom(
                env, new_population[i], base_controller_params, trial,
                fitness_mode, stage_name
            )

            new_fitnesses[i] = custom_fit
            trial += 1

            params = individual_log_payload(new_population[i], fitness_mode, stage_name)
            best_so_far = logger(trial, RUN_ID, params, custom_fit)

            print(f"[GEN {gen}] ind={i+1}/{len(new_population)} fit={custom_fit:.2f}")

        # --- Update population ---
        population = new_population
        fitnesses = new_fitnesses

        # --- Best individual of the generation ---
        if any(f is not None for f in fitnesses):
            best_idx = int(np.nanargmax([f if f is not None else -1e12 for f in fitnesses]))
            gen_best = float(fitnesses[best_idx])
        else:
            gen_best = -1e12
            best_idx = 0

        # --- Single-individual global-best Hall-of-Fame, as in Stage 4b ---
        if gen_best > global_best_fit:
            global_best_fit = gen_best
            global_best = population[best_idx].copy()
            global_best_gen = gen

            # Save new global best for offline evaluation.
            save_hybrid_genome(
                path=os.path.join(
                    "saved_policies",
                    "stage5_answerFitness",
                    f"run{RUN_ID}_gen{gen}_{stage_name}_{fitness_mode}_fit{global_best_fit:.6f}.json",
                ),
                ind=global_best,
                base_controller_params=base_controller_params,
            )


        # Ensure that the global best remains in the population.
        if global_best is not None:
            in_pop = any(
                np.array_equal(ind.cnn_vec, global_best.cnn_vec)
                and np.array_equal(ind.bp_vec, global_best.bp_vec)
                for ind in population
            )
            if not in_pop:
                population[-1] = global_best.copy()
                fitnesses[-1] = global_best_fit

        # --- Robust average fitness through trimmed mean, used for logging ---
        valid_fits = [float(f) for f in fitnesses if f is not None and np.isfinite(float(f))]
        avg_trimmed = trimmed_mean(valid_fits, trim_ratio=0.10) if valid_fits else float("nan")

        print(f"[GEN {gen}] best={gen_best:.2f} avg_trimmed={avg_trimmed:.2f}")

        gen += 1

    # ----- End of run -----
    print("\n" + "=" * 60)
    print(f"[DONE] trials={trial}/{EVAL_BUDGET}, gens={gen-1}")
    print(f"[DONE] global_best={global_best_fit:.2f}")
    print(f"[DONE] CSV: {CSV_PATH}")
    print("=" * 60)

    # --- Save global best genome for offline evaluation ---
    if global_best is not None:
        save_payload = {
            "cnn_weights": global_best.cnn_vec.tolist(),
            "behavior_params": bp_vec_to_dict(global_best.bp_vec),
            "controller_params": base_controller_params,
            "meta": {
                "stage": stage_name,
                "fitness_mode": fitness_mode,
                "run_id": int(RUN_ID),
                "generation": int(global_best_gen),
                "fitness": float(global_best_fit),
                "eval_budget": int(EVAL_BUDGET),
                "pop": int(POP),
                "seed": int(SEED),
                "effective_seed": int(effective_seed),
                "k_repeats": int(K_REPEATS),
                "max_steps": int(MAX_STEPS),
            },
        }

        out_dir = os.path.join("logs", "best_genomes_stage5_answerFitness", fitness_mode)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(
            out_dir,
            f"best_run{RUN_ID}_{stage_name}_{fitness_mode}_fit{global_best_fit:.6f}.json")
        with open(out_path, "w") as f:
            json.dump(save_payload, f, indent=2)

        print(f"[DONE] Saved best genome → {out_path}")
    else:
        print("[WARN] No global_best found; nothing to save.")


if __name__ == "__main__":
    main()
