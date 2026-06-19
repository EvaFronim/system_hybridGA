"""Clean Bayesian Optimization baseline for the Vision-8 search space.

This module runs an Optuna TPE optimizer over the shared Vision-8
behavioral/preprocessing parameter space. Each sampled parameter set 
is tested with a minimal hand-crafted CTE-based controller. Steering 
is computed from the simulator-provided cross-track error signal, while 
throttle is reduced when the estimated local curvature exceeds a threshold.

The goal of this baseline is to provide a clean Bayesian Optimization reference
point that can be compared fairly against the clean GA baseline using the same
environment setup, parameter schema, seeding convention, evaluation budget, and
CSV logging format.
"""


import os
import numpy as np
import optuna
import json


from .common_functions import (
    K_REPEATS,
    SEED,
    RUN_ID,
    EVAL_BUDGET,
    MAX_STEPS,
    PRIMARY_ENV_ID,
    FALLBACK_ENV_IDS,
    ENV_CONF,
    BP_BOUNDS,
    clamp_bp_dict,
    make_logger,
    set_all_seeds,
    reset_env_compat,
    step_env_compat,
    try_make_env,
)


SPACE_NAME = "BO-Vision8"
OPT_NAME = "BO"
N_STARTUP_TRIALS = int(os.environ.get("N_STARTUP_TRIALS", "20"))
LOG_DIR = os.path.join("logs", "runs_bo_vision8")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "trials.csv")


def evaluate_params_clean(
    env,
    params_dict: dict,
    max_steps: int = 1000,
    seed: int | None = None,
) -> float:
    """Evaluate one parameter set using a minimal hand-crafted controller.

    This evaluator does not use any GA, CNN, controller class, or curriculum
    logic. Steering is computed from the environment-reported cross-track error
    signal, while throttle is reduced when the estimated curvature exceeds a
    threshold. Fitness is defined as the cumulative environment reward over the
    episode.
    """
    params = clamp_bp_dict(params_dict)

    kp_off = params["lane_kp_off"]
    kp_head = params["lane_kp_head"]
    kp_curv = params["lane_kp_curv"]
    curv_thr = params["curv_thr_slow"]

    base_throttle = 0.08
    slow_throttle = 0.05

    _, info = reset_env_compat(env, seed=seed)

    total_reward = 0.0
    prev_cte = 0.0
    prev_dcte = 0.0

    for _ in range(max_steps):
        cte = float(info.get("cte", 0.0))
        dcte = cte - prev_cte
        curv_est = abs(dcte - prev_dcte)
        steer = kp_off * cte + kp_head * dcte + kp_curv * curv_est
        steer = max(-1.0, min(1.0, float(steer)))
        throttle = slow_throttle if curv_est >= curv_thr else base_throttle
        action = [steer, throttle]
        _, reward, terminated, truncated, info = step_env_compat(env, action)
        total_reward += float(reward)

        if terminated or truncated:
            break

        prev_dcte = dcte
        prev_cte = cte

    return float(total_reward)

def suggest_vision_params(trial: optuna.Trial) -> dict:
    """Sample one Vision-8 parameter set using Optuna's trial API.

    The search bounds are shared with the GA baseline through BP_BOUNDS so that
    both optimizers operate on the same parameter space.
    """
    return {
        "lane_kp_off":        trial.suggest_float("lane_kp_off",        *BP_BOUNDS[0]),
        "lane_kp_head":       trial.suggest_float("lane_kp_head",       *BP_BOUNDS[1]),
        "lane_kp_curv":       trial.suggest_float("lane_kp_curv",       *BP_BOUNDS[2]),
        "curv_thr_slow":      trial.suggest_float("curv_thr_slow",      *BP_BOUNDS[3]),
        "lane_conf_thr":      trial.suggest_float("lane_conf_thr",      *BP_BOUNDS[4]),
        "canny_scale":        trial.suggest_float("canny_scale",        *BP_BOUNDS[5]),
        "hough_threshold":    trial.suggest_int  ("hough_threshold",     int(BP_BOUNDS[6][0]), int(BP_BOUNDS[6][1])),
        "diag_roi_row_start": trial.suggest_float("diag_roi_row_start", *BP_BOUNDS[7]),
    }

def main() -> None:
    """Run Bayesian Optimization on the clean CTE-PD controller baseline."""
    run_id_int = int(RUN_ID)
    base_seed = int(SEED)
    effective_seed = base_seed + run_id_int

    set_all_seeds(effective_seed)

    print(
        f"[INFO] RUN_ID={run_id_int}, "
        f"BASE_SEED={SEED}, "
        f"EFFECTIVE_SEED={effective_seed}"
    )

    # Create simulator environment.
    env = try_make_env(PRIMARY_ENV_ID, fallback_ids=FALLBACK_ENV_IDS, conf=ENV_CONF)
    if env is None:
        raise RuntimeError("Failed to create the Donkey environment. Start the simulator or verify ENV_CONF.")
    
    logger = make_logger(CSV_PATH, SPACE_NAME, optimizer=OPT_NAME, base_seed=SEED, effective_seed=effective_seed)

    def objective(trial: optuna.Trial) -> float:
        """Evaluate one Optuna trial and log its mean fitness."""
        params = suggest_vision_params(trial)

        # Keep trial numbering 1-based to align with GA trial pairing/logging.
        trial_idx = trial.number + 1

        fitness_values = []
        for repeat_idx in range(K_REPEATS):
            seed = base_seed + run_id_int * 100000 + trial_idx * 100 + repeat_idx
            set_all_seeds(seed)

            fitness = evaluate_params_clean(
                env,
                params,
                max_steps=MAX_STEPS,
                seed=seed,
            )
            fitness_values.append(fitness)

        mean_fitness = float(np.mean(fitness_values))
        logger(trial_idx, run_id_int, params, mean_fitness)
        return mean_fitness

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=effective_seed,
            n_startup_trials=N_STARTUP_TRIALS,
        ),
    )

    try:
        study.optimize(
            objective,
            n_trials=EVAL_BUDGET,
            show_progress_bar=False,
        )
    finally:
        if hasattr(env, "close"):
            env.close()

    print(f"[BO-clean] trials={EVAL_BUDGET}, best_value={study.best_value:.6f}")
    print(f"[BO-clean] best_params={json.dumps(study.best_params, sort_keys=True)}")
    print(f"[BO-clean] CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()