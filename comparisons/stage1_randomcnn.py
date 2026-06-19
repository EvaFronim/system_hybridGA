"""
Stage 1 ablation: frozen CNN with evolved edge-preprocessing parameters.

The genome keeps the same 8-parameter structure used across the full study for
schema consistency. However, in Stages 1-4 the evaluator uses a simplified
edge-image controller that maps preprocessed edge images directly to driving
actions. Therefore, the explicit lane-keeping parameters
(lane_kp_off, lane_kp_head, lane_kp_curv, curv_thr_slow) are logged but do not
directly affect the driving policy in this stage.

Only the preprocessing-related parameters affect the controller input:
lane_conf_thr, canny_scale, hough_threshold, and diag_roi_row_start.

The CNN weights remain frozen during evaluation.
"""

import os, math
import numpy as np

try:
    import torch
except Exception:
    torch = None

from scripts.main_ga import try_make_env

from src.controller import Controller, ControllerConfig


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
    controller_cfg_genome,
    make_logger,
    mutate_bp_vec,
    bp_vec_to_dict,
    crossover_blend_bp_vec,
    evaluate_with_frozen_cnn,
    rand_bp_vec,
    set_all_seeds,
)

SPACE_NAME = "Vision4_FrozenCNN_EdgeController"
LOG_DIR = os.path.join("logs", "runs_ga_stage1_randomcnn")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "trials.csv")

def _initialize_frozen_controller():
    """
    Attempts to initialize the Controller in a frozen/random state.
    
    This helper handles different versions of the Controller API, 
    ensuring that no pre-trained weights or genomes are loaded.
    
    Returns:
        Controller: An initialized instance of the vehicle controller.
        
    Raises:
        RuntimeError: If the controller cannot be initialized with the given API variants.
    """
    cfg = ControllerConfig()

    # Constructor variant 1: Controller accepts load_weights directly.
    try:
        return Controller(config=cfg, load_weights=False)
    except TypeError:
        pass

    # Constructor variant 2: ControllerConfig exposes load/genome options.
    try:
        if hasattr(cfg, "load_weights"):
            setattr(cfg, "load_weights", False)
        if hasattr(cfg, "genome"):
            setattr(cfg, "genome", None)

        return Controller(config=cfg)

    except Exception as e:
        raise RuntimeError(
            "Controller initialization failed. Ensure Controller can be "
            f"created as a frozen/random CNN without requiring a genome. Original error: {e}"
        )

def main():
    """
    Executes the Stage 1 GA ablation loop.
    
    The process involves:
    1. Initializing a frozen CNN controller.
    2. Generating an initial population of preprocessing parameters.
    3. Evaluating fitness through repeated rollouts in the Donkey Car simulator.
    4. Iteratively evolving the population using tournament selection, 
       crossover, and mutation.
    """
    if torch is None:
        raise ImportError("PyTorch is required to run Stage 1.")

    # Initialize environment and random number generators
    run_id_int = int(RUN_ID)
    effective_seed = int(SEED) + run_id_int
    set_all_seeds(effective_seed)

    print(
        f"[INFO] RUN_ID={run_id_int}, "
        f"BASE_SEED={SEED}, "
        f"EFFECTIVE_SEED={effective_seed}"
    )

    if torch is not None:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    rng = np.random.default_rng(effective_seed)

    # Create simulator environment.
    env = try_make_env(PRIMARY_ENV_ID, fallback_ids=FALLBACK_ENV_IDS, conf=ENV_CONF)
    if env is None:
        raise RuntimeError(
            "Cannot create Donkey env. Ensure the simulator is running at the "
            "configured host/port or adjust ENV_CONF."
        )

    try:
        controller = _initialize_frozen_controller()

        # Freeze model parameters and switch to evaluation mode.
        if hasattr(controller, "model"):
            controller.model.eval()

            for param in controller.model.parameters():
                param.requires_grad_(False)

        logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)

        pop = max(1, int(POP))
        eval_budget = max(1, int(EVAL_BUDGET))
        max_generations = max(1, math.ceil(eval_budget / pop))

        # Initialize population.
        population = [rand_bp_vec(rng) for _ in range(pop)]
        fitnesses = [None] * pop

        trial = 0
        best_so_far = -1e12

        print(
            f"[INFO] Stage 1 started: pop={pop}, budget={eval_budget}, "
            f"k_repeats={K_REPEATS}, max_steps={MAX_STEPS}, seed={SEED}, run_id={RUN_ID}"
        )

        # Evaluate initial population.
        for i in range(pop):
            if trial >= eval_budget:
                break

            params = bp_vec_to_dict(population[i])
            fits = []

            for r in range(K_REPEATS):
                rollout_seed = SEED + run_id_int * 100000 + (trial + 1) * 100 + r
                set_all_seeds(rollout_seed)

                if hasattr(controller, "reset_turn_stabilizer"):
                    controller.reset_turn_stabilizer()

                fitness = evaluate_with_frozen_cnn(
                    env,
                    controller,
                    params,
                    max_steps=MAX_STEPS,
                    seed=rollout_seed,
                )
                fits.append(fitness)

            fit = float(np.mean(fits))
            fit_std = float(np.std(fits))

            trial += 1
            best_so_far = logger(trial, RUN_ID, params, fit)
            fitnesses[i] = fit

            print(
                f"[GEN 0] ind={i + 1}/{pop} "
                f"fit={fit:.2f} ±{fit_std:.2f} best={best_so_far:.2f}"
            )

        # Evolution loop.
        gen = 1

        while trial < eval_budget and gen <= max_generations:
            print(f"\n[INFO] Generation {gen}/{max_generations}")

            def tournament():
                """Select one parent using tournament selection."""
                idxs = rng.choice(
                    len(population),
                    size=min(3, len(population)),
                    replace=False,
                )
                best_idx = max(idxs, key=lambda j: fitnesses[j])
                return population[best_idx]

            new_offspring = []

            while len(new_offspring) < pop and trial < eval_budget:
                parent_1 = tournament()
                parent_2 = tournament()

                child = crossover_blend_bp_vec(parent_1, parent_2, rng, alpha=0.2)
                child = mutate_bp_vec(child, rng, sigma_scale=0.10, p=0.25)

                params = bp_vec_to_dict(child)
                fits = []

                for r in range(K_REPEATS):
                    rollout_seed = SEED + run_id_int * 100000 + (trial + 1) * 100 + r
                    set_all_seeds(rollout_seed)

                    if hasattr(controller, "reset_turn_stabilizer"):
                        controller.reset_turn_stabilizer()

                    fitness = evaluate_with_frozen_cnn(
                        env,
                        controller,
                        params,
                        max_steps=MAX_STEPS,
                        seed=rollout_seed,
                    )
                    fits.append(fitness)

                fit = float(np.mean(fits))
                fit_std = float(np.std(fits))

                print(f"[DEBUG] trial={trial + 1} fit={fit:.2f} ±{fit_std:.2f}")

                trial += 1
                best_so_far = logger(trial, RUN_ID, params, fit)
                new_offspring.append((child, fit))

            all_individuals = list(zip(population, fitnesses)) + new_offspring
            all_individuals = [
                (genome, fitness)
                for genome, fitness in all_individuals
                if fitness is not None
            ]

            all_individuals.sort(key=lambda item: item[1], reverse=True)
            all_individuals = all_individuals[:pop]

            population = [genome for genome, _ in all_individuals]
            fitnesses = [fitness for _, fitness in all_individuals]

            print(
                f"[GEN {gen}] best={fitnesses[0]:.2f} "
                f"avg={float(np.mean(fitnesses)):.2f}"
            )

            gen += 1

        print(f"\n[DONE] trials={trial}/{eval_budget}, pop={pop}, gens={gen - 1}")
        print(f"[DONE] best_so_far={best_so_far:.2f}")
        print(f"[DONE] CSV: {CSV_PATH}")

    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
