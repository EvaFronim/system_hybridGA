"""Clean Genetic Algorithm baseline for the Vision-8 search space.

This module runs a simple GA over the shared Vision-8 behavioral/preprocessing
parameter space. Εach candidate parameter set is tested with a minimal hand-crafted
CTE-based controller. Steering is computed from the simulator-provided
cross-track error signal, while throttle is reduced when the estimated local
curvature exceeds a threshold.

The goal of this baseline is not to produce the strongest driving policy, but
to provide a clean GA reference point that can be compared fairly against the
Bayesian Optimization baseline using the same environment setup, parameter
schema, seeding convention, and CSV logging format.
"""

import os
import math
import numpy as np

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
    clamp_bp_dict,
    make_logger,
    set_all_seeds,
    rand_bp_vec,
    mutate_bp_vec,
    crossover_blend_bp_vec,
    bp_vec_to_dict,
    reset_env_compat,
    step_env_compat,
    try_make_env,
)


SPACE_NAME = "Vision-8"
LOG_DIR = os.path.join("logs", "runs_ga_vision8")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "trials.csv")


def evaluate_params_clean(env,params_dict: dict,max_steps: int = 1000,seed: int | None = None) -> float:
    """Evaluate one Vision-8 parameter set with a hand-crafted CTE controller.

    Only the first four parameters affect this clean baseline:
    (lane_kp_off, lane_kp_head, lane_kp_curv, cuev_thr_slow)

    The remaining Vision-8 parameters are still accepted and logged through the
    shared parameter schema, but they do not affect this evaluator because no
    vision preprocessing or CNN controller is used.

    Fitness is the cumulative simulator reward over one rollout.
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

        # Estimate heading/curvature indirectly from consecutive CTE changes.
        cte = float(info.get("cte", 0.0))
        dcte = cte - prev_cte
        curv_est = abs(dcte - prev_dcte)

        # CTE-PD-style steering with an additional curvature term.
        steer = kp_off * cte + kp_head * dcte + kp_curv * curv_est
        steer = max(-1.0, min(1.0, float(steer)))

        # Slow down on sharper estimated turns.
        throttle = slow_throttle if curv_est >= curv_thr else base_throttle
        action = [steer, throttle]

        _, reward, terminated, truncated, info = step_env_compat(env, action)

        total_reward += float(reward)
        if terminated or truncated:
            break

        prev_dcte = dcte
        prev_cte = cte

    return float(total_reward)

def main() -> None:
    """Run the clean GA baseline within the configured evaluation budget."""

    # Derive a deterministic run-specific seed.
    run_id_int = int(RUN_ID)
    base_seed = int(SEED)
    effective_seed = base_seed + run_id_int

    set_all_seeds(effective_seed)

    print(
        f"[INFO] RUN_ID={run_id_int}, "
        f"BASE_SEED={base_seed}, "
        f"EFFECTIVE_SEED={effective_seed}"
    )

    rng = np.random.default_rng(effective_seed)  # This RNG controls GA sampling, crossover, mutation, and tournament selection.

    env = try_make_env(
        PRIMARY_ENV_ID,
        fallback_ids=FALLBACK_ENV_IDS,
        conf=ENV_CONF,
    )
    if env is None:
        raise RuntimeError(
            "Failed to create the Donkey environment. Start the simulator "
            "GUI on port 9091 or verify ENV_CONF."
        )
    logger = make_logger(CSV_PATH, SPACE_NAME, optimizer="GA", base_seed=base_seed, effective_seed=effective_seed)
    pop = max(1, int(POP))
    max_generations = max(1, math.ceil(EVAL_BUDGET / pop))
    population = [rand_bp_vec(rng) for _ in range(pop)]
    fitnesses = [None] * pop
    trial = 0

    try:
        for idx in range(pop):
            if trial >= EVAL_BUDGET:
                break

            params = bp_vec_to_dict(population[idx])
            fitness_values = []
            for repeat_idx in range(K_REPEATS):
                seed = base_seed + run_id_int * 100000 + (trial + 1) * 100 + repeat_idx
                set_all_seeds(seed)
                fitness_values.append(
                    evaluate_params_clean(
                        env,
                        params,
                        max_steps=MAX_STEPS,
                        seed=seed,
                    )
                )

            mean_fitness = float(np.mean(fitness_values))
            trial += 1
            logger(trial, run_id_int, params, mean_fitness)
            fitnesses[idx] = mean_fitness

        generation = 1
        while trial < EVAL_BUDGET and generation <= max_generations:

            def tournament() -> np.ndarray:
                """Select one parent with tournament selection (k=3)."""
                candidate_indices = rng.choice(
                    len(population),
                    size=min(3, len(population)),
                    replace=False,
                )
                best_index = max(candidate_indices, key=lambda j: fitnesses[j])
                return population[best_index]

            new_offspring = []

            while len(new_offspring) < pop and trial < EVAL_BUDGET:
                parent_a = tournament()
                parent_b = tournament()

                child = crossover_blend_bp_vec(parent_a, parent_b, rng, alpha=0.2)
                child = mutate_bp_vec(child, rng, sigma_scale=0.10, p=0.25)

                params = bp_vec_to_dict(child)

                fitness_values = []
                for repeat_idx in range(K_REPEATS):
                    seed = base_seed + run_id_int * 100000 + (trial + 1) * 100 + repeat_idx
                    set_all_seeds(seed)
                    fitness_values.append(
                        evaluate_params_clean(
                            env,
                            params,
                            max_steps=MAX_STEPS,
                            seed=seed,
                        )
                    )

                mean_fitness = float(np.mean(fitness_values))
                trial += 1
                logger(trial, run_id_int, params, mean_fitness)
                new_offspring.append((child, mean_fitness))

            all_individuals = list(zip(population, fitnesses)) + new_offspring
            all_individuals = [
                (vector, fitness)
                for vector, fitness in all_individuals
                if fitness is not None
            ]
            all_individuals.sort(key=lambda item: item[1], reverse=True)
            all_individuals = all_individuals[:pop]
            population = [vector for vector, _ in all_individuals]
            fitnesses = [fitness for _, fitness in all_individuals]
            generation += 1
    finally:
        if hasattr(env, "close"):
            env.close()

    print(
        f"[GA-clean] trials={trial}/{EVAL_BUDGET}, "
        f"population={pop}, max_generations={max_generations}"
    )
    print(f"[GA-clean] CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()