#!/usr/bin/env python3
"""Neutral post-hoc evaluation of Stage 5 best-per-run policies.

After the Stage 5 fitness ablation training (stage5_fitnessAblation.py),
this script loads the saved best genome from each run and evaluates every
policy under three independent protocols:

  1. F3 fitness score         (eval_new_fitness_f3)
  2. Clean rollout metrics    (clean_eval)
  3. Simulator telemetry      (clean_telemetry_eval, assisted_eval)

Results are written to a single CSV for downstream statistical analysis.
Each row corresponds to one saved policy (expected: 15 runs × 4 modes = 60).
"""

import os
import glob
import json
import csv
import types
import re

import numpy as np

from comparisons import stage5_fitnessAblation

from comparisons.stage5_fitnessAblation import (
    MockCurriculum,
    MockConfig,
    SEED,
)

from comparisons.common_functions import (
    set_all_seeds,
    PARAM_KEYS,
    try_make_env,
    PRIMARY_ENV_ID,
    FALLBACK_ENV_IDS,
    ENV_CONF,
    bp_vec_to_dict,
)


# ------ Config --------------

# Input: best-per-run policies
BEST_GENOMES_DIR = "logs/best_genomes_stage5_answerFitness"

# Output CSV
OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    "logs/eval_correlation/policies_eval_BEST_PER_RUN.csv",
)

# Evaluation repeats
REPEATS = 5

MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))

def load_policy_with_metadata(path: str):
    """Load a saved genome JSON and extract vectors plus experiment metadata.

    Behavior-parameter loading supports both dict and list formats for
    backward compatibility with older saved genomes.
    Metadata fields (run_id, fitness_mode) fall back to filename parsing
    if absent from the JSON meta block.

    Returns:
    cnn_vec, bp_vec, controller_params, metadata dict
    """
    with open(path, "r") as f:
        data = json.load(f)

    cnn_vec = np.asarray(data["cnn_weights"], dtype=np.float32)

    bp_raw = data["behavior_params"]
    if isinstance(bp_raw, dict):
        bp_vec = np.asarray([bp_raw[k] for k in PARAM_KEYS], dtype=np.float32)
    else:
        bp_vec = np.asarray(bp_raw, dtype=np.float32)

    controller_params = data.get("controller_params", {}) or {}

    meta = data.get("meta", {})

    filename = os.path.basename(path)
    run_match = re.search(r'run(\d+)', filename)
    mode_match = re.search(r'_(f\d+)_', filename)

    fitness_mode = meta.get('fitness_mode') or (
        mode_match.group(1) if mode_match else 'UNKNOWN'
    )

    metadata = {
        'run_id': int(meta.get('run_id')) if meta.get('run_id') is not None else (
            int(run_match.group(1)) if run_match else -1
        ),
        'fitness_mode': str(fitness_mode).upper(),
        'generation': meta.get('generation', -1),
        'training_fitness': meta.get('fitness', 0.0),
        'stage_name': meta.get('stage', 'Basic'),
    }

    return cnn_vec, bp_vec, controller_params, metadata

# ------- Evaluation Functions ------

def eval_new_fitness_f3(env, cnn_vec, bp_vec, controller_params, 
                        stage_name="Basic", repeats=REPEATS):
    """Evaluate a policy with the F3 (full system) fitness as a neutral comparator.

    Applied uniformly to all policies regardless of training mode, so scores
    are directly comparable across F0–F3 runs.

    Uses a fixed seed family (2000 + r) that is disjoint from both the
    training seeds and the other evaluation protocols.
    """
    bp_dict = bp_vec_to_dict(bp_vec)
    
    curriculum = MockCurriculum(stage_name)
    cfg = stage5_fitnessAblation.AdaptiveConfig()
    hp = cfg.hparams
    for k, v in bp_dict.items():
        if hasattr(hp, k):
            setattr(hp, k, v)

    all_metrics = []
    all_actions_runs = []

    for r in range(repeats):
        seed = 2000 + r
        set_all_seeds(seed)

        mock_genome = types.SimpleNamespace(
            cnn_weights=cnn_vec.tolist(),
            behavior_params=bp_vec.tolist(),
            controller_params=controller_params,
        )

        metrics, actions = stage5_fitnessAblation.improved_throttle_control_with_slow_penalty(
            env=env, genome=mock_genome, config=cfg,
            curriculum=curriculum, track_actions=True,
            action_sample_stride=5,
        )

        if hasattr(metrics, "is_donut") and getattr(metrics, "is_donut", False):
            metrics.distance = 0.0

        all_metrics.append(metrics)
        all_actions_runs.append(actions)

    fit_f3 = stage5_fitnessAblation.compute_fitness(
        all_metrics, all_actions_runs,
        mode="f3", stage_name=stage_name,
    )

    return float(fit_f3)


class InfoLoggingEnv:
    """Thin environment wrapper that records the info dict from every step.

    Used by clean_telemetry_eval and assisted_eval to access simulator
    telemetry (pos, cte, speed, hit, forward_vel) without modifying the
    underlying evaluation functions.
    """
    def __init__(self, env):
        self.env = env
        self.infos = []

    def reset(self, *args, **kwargs):
        self.infos = []
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.infos.append(info)
        return obs, reward, done, info

    def __getattr__(self, name):
        # forward everything else to the real env
        return getattr(self.env, name)
    
def clean_eval(env, cnn_vec, bp_vec, controller_params,
               stage_name="Basic", repeats=REPEATS):
    """
    Clean policy evaluation without the F3 rule-assisted steering/throttle corrections.
    This evaluates the genome using stage5.clean_policy_rollout, so it is better
    suited for comparing which fitness objective produced better raw policies.
    """
    bp_dict = bp_vec_to_dict(bp_vec)
    curriculum = stage5_fitnessAblation.MockCurriculum(stage_name)
    config = stage5_fitnessAblation.MockConfig(bp_dict)

    distances = []
    stabilities = []
    efficiencies = []
    explorations = []
    consistencies = []
    lane_confs = []
    turn_handlings = []
    speed_consistencies = []
    slow_rates = []
    offlane_rates = []
    all_clean_actions = []

    for r in range(repeats):
        # Use same seed family, so raw evaluations are comparable.
        s = SEED + 2_000_000 + r
        set_all_seeds(s)

        genome_like = types.SimpleNamespace(
            cnn_weights=cnn_vec.tolist(),
            behavior_params=bp_vec.tolist(),
            controller_params=controller_params,
        )

        metrics, actions = stage5_fitnessAblation.clean_policy_rollout(
            env=env,
            genome=genome_like,
            config=config,
            curriculum=curriculum,
            track_actions=True,
            action_sample_stride=5,
        )

        if hasattr(metrics, "is_donut") and getattr(metrics, "is_donut", False):
            metrics.distance = 0.0

        distances.append(float(metrics.distance))
        stabilities.append(float(metrics.stability))
        efficiencies.append(float(metrics.efficiency))
        explorations.append(float(metrics.exploration))
        consistencies.append(float(metrics.consistency))
        lane_confs.append(float(metrics.lane_conf))
        turn_handlings.append(float(metrics.turn_handling))
        speed_consistencies.append(float(getattr(metrics, "speed_consistency", metrics.efficiency)))
        slow_rates.append(float(getattr(metrics, "slow_rate", 0.0)))
        offlane_rates.append(float(getattr(metrics, "offlane_rate", 0.0)))
        all_clean_actions.append(actions)

        metrics_list_objects = [
            types.SimpleNamespace(
                distance=d, stability=s, efficiency=e, exploration=x, consistency=c,
                lane_conf=l, turn_handling=t, speed_consistency=sc, slow_rate=sr, offlane_rate=orate
            ) for d, s, e, x, c, l, t, sc, sr, orate in zip(
                distances, stabilities, efficiencies, explorations, consistencies,
                lane_confs, turn_handlings, speed_consistencies, slow_rates, offlane_rates
            )
        ]

    clean_f0 = stage5_fitnessAblation.compute_fitness(
        metrics_list_objects,
        actions_list=all_clean_actions,
        mode="f0",
        stage_name=stage_name,
    )

    clean_f1 = stage5_fitnessAblation.compute_fitness(
        metrics_list_objects,
        actions_list=all_clean_actions,
        mode="f1",
        stage_name=stage_name,
    )

    clean_f2 = stage5_fitnessAblation.compute_fitness(
        metrics_list_objects,
        actions_list=all_clean_actions,
        mode="f2",
        stage_name=stage_name,
    )

    clean_f3 = stage5_fitnessAblation.compute_fitness(
        metrics_list_objects, 
        actions_list=all_clean_actions, 
        mode="f3", 
        stage_name=stage_name)

    return {
        "clean_distance": float(np.mean(distances)),
        "clean_stability": float(np.mean(stabilities)),
        "clean_efficiency": float(np.mean(efficiencies)),
        "clean_exploration": float(np.mean(explorations)),
        "clean_consistency": float(np.mean(consistencies)),
        "clean_lane_conf": float(np.mean(lane_confs)),
        "clean_turn_handling": float(np.mean(turn_handlings)),
        "clean_speed_consistency": float(np.mean(speed_consistencies)),
        "clean_slow_rate": float(np.mean(slow_rates)),
        "clean_offlane_rate": float(np.mean(offlane_rates)),
        "clean_f0_score": float(clean_f0),
        "clean_f1_score": float(clean_f1),
        "clean_f2_score": float(clean_f2),
        "clean_f3_score": float(clean_f3),
    }

def clean_telemetry_eval(env, cnn_vec, bp_vec, controller_params,
                         stage_name="Basic", repeats=REPEATS):
    """
    Raw/clean telemetry evaluation.

    This evaluates the saved genome with clean_policy_rollout,
    without assisted steering/throttle corrections.

    Use these metrics to answer:
        Which fitness mode produces the best raw driver?
    """
    bp_dict = bp_vec_to_dict(bp_vec)
    curriculum = stage5_fitnessAblation.MockCurriculum(stage_name)
    config = stage5_fitnessAblation.MockConfig(bp_dict)

    distances_pos = []
    crashes = []
    mean_speeds = []
    forward_progresses = []
    mean_abs_ctes = []
    steps_survived = []

    log_env = InfoLoggingEnv(env)

    for r in range(repeats):
        # Use same seed family as clean_eval, so raw evaluations are comparable.
        s = SEED + 2_000_000 + r
        set_all_seeds(s)

        genome_like = types.SimpleNamespace(
            cnn_weights=cnn_vec.tolist(),
            behavior_params=bp_vec.tolist(),
            controller_params=controller_params,
        )

        metrics, _ = stage5_fitnessAblation.clean_policy_rollout(
            env=log_env,
            genome=genome_like,
            config=config,
            curriculum=curriculum,
            track_actions=False,
            action_sample_stride=5,
        )

        infos = log_env.infos

        dist = 0.0
        prev = None
        speeds = []
        hit_events = 0
        abs_ctes = []
        forward_progress = 0.0
        prev_hit = "none"

        for info in infos:
            # CTE
            cte = info.get("cte", None)
            if cte is not None:
                abs_ctes.append(abs(float(cte)))

            # Forward progress from simulator telemetry
            fv = info.get("forward_vel", None)
            if fv is not None:
                forward_progress += max(float(fv), 0.0)

            # Position-based travelled distance
            pos = info.get("pos", None)
            if pos is not None:
                x, _, z = pos
                if prev is not None:
                    dx = x - prev[0]
                    dz = z - prev[1]
                    dist += float((dx * dx + dz * dz) ** 0.5)
                prev = (x, z)

            # Speed
            if "speed" in info:
                speeds.append(float(info["speed"]))

            # Hit events: count none -> collision transitions
            hit = info.get("hit", "none")
            if isinstance(hit, str) and hit != "none" and prev_hit == "none":
                hit_events += 1
            prev_hit = hit

        # Same donut guard as the other evaluations
        if hasattr(metrics, "is_donut") and getattr(metrics, "is_donut", False):
            dist = 0.0
            forward_progress = 0.0

        distances_pos.append(float(dist))
        crashes.append(float(hit_events))
        mean_speeds.append(float(np.mean(speeds)) if speeds else 0.0)
        forward_progresses.append(float(forward_progress))
        mean_abs_ctes.append(float(np.mean(abs_ctes)) if abs_ctes else np.nan)
        steps_survived.append(float(len(infos)))

    return {
        "clean_distance_pos": float(np.mean(distances_pos)),
        "clean_crashes": float(np.mean(crashes)),
        "clean_mean_speed": float(np.mean(mean_speeds)),
        "clean_forward_progress": float(np.mean(forward_progresses)),
        "clean_mean_abs_cte": float(np.nanmean(mean_abs_ctes)) if not np.all(np.isnan(mean_abs_ctes)) else np.nan,
        "clean_steps_survived": float(np.mean(steps_survived)),
    }

def assisted_eval(env, cnn_vec, bp_vec, controller_params,
                 stage_name="Basic", repeats=REPEATS):
    """assisted evaluation from env telemetry: traveled distance (pos), lap_count, hit events, mean speed.
    Each policy was evaluated on the same fixed set of evaluation seeds."""
    bp_dict = bp_vec_to_dict(bp_vec)
    curriculum = stage5_fitnessAblation.MockCurriculum(stage_name)
    config = stage5_fitnessAblation.MockConfig(bp_dict)

    distances = []
    laps = []
    crashes = []
    mean_speeds = []
    forward_progresses = []
    mean_abs_ctes = []


    log_env = InfoLoggingEnv(env)  # wrap ONCE

    for r in range(repeats):
        s = SEED + 1_000_000 + r
        set_all_seeds(s)

        genome_like = types.SimpleNamespace(
            cnn_weights=cnn_vec.tolist(),
            behavior_params=bp_vec.tolist(),
            controller_params=controller_params,
        )

        metrics, _ = stage5_fitnessAblation.improved_throttle_control_with_slow_penalty(
            env=log_env, genome=genome_like, config=config,
            curriculum=curriculum, track_actions=False,
            action_sample_stride=5,
        )
        infos = log_env.infos

        # distance from pos (x,z)
        dist = 0.0
        prev = None
        speeds = []
        hit_events = 0
        lap_count = 0
        abs_ctes = []
        forward_progress = 0.0
        prev_hit = "none"

        for info in infos:
            cte = info.get("cte", None)
            if cte is not None:
                abs_ctes.append(abs(float(cte)))

            fv = info.get("forward_vel", None)
            if fv is not None:
                forward_progress += max(float(fv), 0.0)

            pos = info.get("pos")
            if pos is not None:
                x, _, z = pos
                if prev is not None:
                    dx = x - prev[0]
                    dz = z - prev[1]
                    dist += float((dx*dx + dz*dz) ** 0.5)
                prev = (x, z)

            if "speed" in info:
                speeds.append(float(info["speed"]))

            hit = info.get("hit", "none")
            # count hit EVENT (none -> something)
            if isinstance(hit, str) and hit != "none" and prev_hit == "none":
                hit_events += 1
            prev_hit = hit

            if "lap_count" in info:
                lap_count = int(info["lap_count"])

        # donut guard
        if hasattr(metrics, "is_donut") and getattr(metrics, "is_donut", False):
            dist = 0.0

        mean_abs_cte = float(np.mean(abs_ctes)) if abs_ctes else 0.0

        forward_progresses.append(float(forward_progress))
        mean_abs_ctes.append(float(mean_abs_cte))
        distances.append(dist)
        laps.append(float(lap_count))
        crashes.append(float(hit_events))
        mean_speeds.append(float(np.mean(speeds)) if speeds else 0.0)

    return (
        float(np.mean(distances)),
        float(np.mean(laps)),
        float(np.mean(crashes)),
        float(np.mean(mean_speeds)),
        float(np.mean(forward_progresses)),
        float(np.mean(mean_abs_ctes)),
    )


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    env = try_make_env(
        PRIMARY_ENV_ID,
        conf=ENV_CONF,
        fallback_ids=FALLBACK_ENV_IDS,
    )
    if env is None:
        raise RuntimeError("Cannot create Donkey env for eval_correlation")

    # Collect all best-per-run policies
    all_paths = []
    
    for mode in ['f0', 'f1', 'f2', 'f3']:
        mode_dir = os.path.join(BEST_GENOMES_DIR, mode)
        if os.path.isdir(mode_dir):
            for p in glob.glob(os.path.join(mode_dir, "best_*.json")):
                all_paths.append(p)
    
    all_paths.sort()
    
    print(f"[INFO] Found {len(all_paths)} best-per-run policies")
    
    # Group by mode for reporting
    mode_counts = {}
    for p in all_paths:
        if '/f0/' in p:
            mode_counts['F0'] = mode_counts.get('F0', 0) + 1
        elif '/f1/' in p:
            mode_counts['F1'] = mode_counts.get('F1', 0) + 1
        elif '/f2/' in p:
            mode_counts['F2'] = mode_counts.get('F2', 0) + 1
        elif '/f3/' in p:
            mode_counts['F3'] = mode_counts.get('F3', 0) + 1
    
    print(f"[INFO] Counts per mode: {mode_counts}")
    
    if not all_paths:
        print(f"[ERROR] No policies found in {BEST_GENOMES_DIR}")
        print(f"[ERROR] Make sure stage5 has saved best genomes there")
        return

    # Evaluate and write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy_id",
                "run_id",
                "fitness_mode",
                "generation",
                "training_fitness",
                "fitness_f3",
                "assisted_distance",
                "assisted_lap",
                "assisted_crashes",
                "assisted_mean_speed",
                "assisted_forward_progress",
                "assisted_mean_abs_cte",
                "clean_distance_pos",
                "clean_crashes",
                "clean_mean_speed",
                "clean_forward_progress",
                "clean_mean_abs_cte",
                "clean_steps_survived",
                "clean_distance",
                "clean_stability",
                "clean_efficiency",
                "clean_exploration",
                "clean_consistency",
                "clean_lane_conf",
                "clean_turn_handling",
                "clean_speed_consistency",
                "clean_slow_rate",
                "clean_offlane_rate",
                "clean_f0_score",
                "clean_f1_score",
                "clean_f2_score",
                "clean_f3_score"
            ],
        )
        writer.writeheader()

        for i, path in enumerate(all_paths):
            fname = os.path.basename(path)
            policy_id = os.path.splitext(fname)[0]

            print(f"[EVAL {i+1}/{len(all_paths)}] {policy_id}")

            cnn_vec, bp_vec, controller_params, metadata = load_policy_with_metadata(path)

            stage_name = metadata.get("stage_name", "Basic")

            fitness_f3 = eval_new_fitness_f3(
                env, cnn_vec, bp_vec, controller_params,
                stage_name=stage_name
            )

            assisted_distance, assisted_lap, assisted_crashes, assisted_mean_speed, assisted_forward_progress, assisted_mean_abs_cte = assisted_eval(
                env, cnn_vec, bp_vec, controller_params,
                stage_name=stage_name
            )

            clean_telemetry = clean_telemetry_eval(
                env, cnn_vec, bp_vec, controller_params,
                stage_name=stage_name
            )

            clean_metrics = clean_eval(
                env, cnn_vec, bp_vec, controller_params,
                stage_name=stage_name
            )


            # Write row
            writer.writerow({
                "policy_id": policy_id,
                "run_id": metadata['run_id'],
                "fitness_mode": metadata['fitness_mode'],
                "generation": metadata['generation'],
                "training_fitness": metadata['training_fitness'],
                "fitness_f3": fitness_f3,
                "assisted_distance": assisted_distance,
                "assisted_lap": assisted_lap,
                "assisted_crashes": assisted_crashes,
                "assisted_mean_speed": assisted_mean_speed,
                "assisted_forward_progress": assisted_forward_progress,
                "assisted_mean_abs_cte": assisted_mean_abs_cte,
                "clean_distance_pos": clean_telemetry["clean_distance_pos"],
                "clean_crashes": clean_telemetry["clean_crashes"],
                "clean_mean_speed": clean_telemetry["clean_mean_speed"],
                "clean_forward_progress": clean_telemetry["clean_forward_progress"],
                "clean_mean_abs_cte": clean_telemetry["clean_mean_abs_cte"],
                "clean_steps_survived": clean_telemetry["clean_steps_survived"],
                "clean_distance": clean_metrics["clean_distance"],
                "clean_stability": clean_metrics["clean_stability"],
                "clean_efficiency": clean_metrics["clean_efficiency"],
                "clean_exploration": clean_metrics["clean_exploration"],
                "clean_consistency": clean_metrics["clean_consistency"],
                "clean_lane_conf": clean_metrics["clean_lane_conf"],
                "clean_turn_handling": clean_metrics["clean_turn_handling"],
                "clean_speed_consistency": clean_metrics["clean_speed_consistency"],
                "clean_slow_rate": clean_metrics["clean_slow_rate"],
                "clean_offlane_rate": clean_metrics["clean_offlane_rate"],
                "clean_f0_score": clean_metrics["clean_f0_score"],
                "clean_f1_score": clean_metrics["clean_f1_score"],
                "clean_f2_score": clean_metrics["clean_f2_score"],
                "clean_f3_score": clean_metrics["clean_f3_score"],
            })

    env.close()
    print(f"\n[DONE] Wrote balanced CSV to: {OUTPUT_CSV}")
    print(f"[DONE] Expected n=15 per mode (total ~60)")


if __name__ == "__main__":
    main()