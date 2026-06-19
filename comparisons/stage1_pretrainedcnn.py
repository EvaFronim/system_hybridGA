"""
Stage 1b ablation: pretrained frozen CNN with evolved edge-preprocessing parameters.

This stage uses the same 8-parameter genome schema as the rest of the ablation
study for logging and comparison consistency. However, the simplified evaluator
used in Stages 1-4 maps preprocessed edge images directly to driving actions
through Controller.predict(edge_img).

Therefore, the explicit lane-keeping parameters
(lane_kp_off, lane_kp_head, lane_kp_curv, curv_thr_slow) are logged but do not
directly affect the driving policy in this stage.

Only the preprocessing-related parameters affect the controller input:
lane_conf_thr, canny_scale, hough_threshold, and diag_roi_row_start.

Unlike Stage 1_randomCNN, this script requires valid pretrained weights and will 
explicitly fail if they are not provided, ensuring experimental integrity.
"""

import os, math
from typing import List

import numpy as np

try:
    import torch
except Exception:
    torch = None

from src.controller import Controller, ControllerConfig


# Import env factory (same as baseline for fairness)
from scripts.main_ga import try_make_env

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
    _load_best_genome_from_files,
)


SPACE_NAME = "Vision4_FrozenPretrainedCNN_EdgeController"

LOG_DIR = os.path.join("logs", "runs_ga_stage1_pretrainedcnn")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "trials.csv")



def main():
    """Run the Stage 1b GA ablation with a pretrained frozen CNN edge-image controller."""
    if torch is None:
        raise ImportError("PyTorch is required to run Stage 1b.")

    pretrained_pth = os.environ.get("PRETRAINED_CNN_PATH", None)
    pretrained_npy = os.environ.get("PRETRAINED_CNN_NPY", None)
    pretrained_json = os.environ.get("PRETRAINED_CNN_JSON", None)

    # Validate paths before starting the simulator
    for path in [p for p in [pretrained_pth, pretrained_npy, pretrained_json] if p]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required weights file missing: {path}")

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

    env = try_make_env(PRIMARY_ENV_ID, fallback_ids=FALLBACK_ENV_IDS, conf=ENV_CONF)
    if env is None:
        raise RuntimeError(
            "Cannot create Donkey env. Ensure the simulator is running at the "
            "configured host/port or adjust ENV_CONF."
        )

    try:
        controller = None

        # Option A: load pretrained CNN through a genome-like object.
        genome = _load_best_genome_from_files(pretrained_npy, pretrained_json)

        if genome is not None:
            print(f"[INFO] Using pretrained genome from npy={pretrained_npy} json={pretrained_json}")

            init_errors: List[str] = []

            constructor_attempts = [
                lambda: Controller(genome=genome, config=ControllerConfig(), load_weights=True),
                lambda: Controller(genome=genome, load_weights=True),
                lambda: controller_cfg_genome(genome),
            ]

            for build_controller in constructor_attempts:
                try:
                    controller = build_controller()
                    break
                except Exception as e:
                    init_errors.append(str(e))
                    controller = None

            if controller is None:
                raise RuntimeError(
                    "Controller initialization with pretrained genome failed. "
                    "Tried multiple constructor signatures. Errors: "
                    + " | ".join(init_errors)
                )

        # Option B: load pretrained CNN from a PyTorch state_dict.
        elif pretrained_pth:
            cfg = ControllerConfig()

            try:
                controller = Controller(config=cfg, load_weights=False)
            except TypeError:
                if hasattr(cfg, "load_weights"):
                    setattr(cfg, "load_weights", False)
                if hasattr(cfg, "genome"):
                    setattr(cfg, "genome", None)
                controller = Controller(config=cfg)

            if not hasattr(controller, "model"):
                raise RuntimeError(
                    "Controller does not expose a `.model` attribute, so PyTorch "
                    "pretrained weights cannot be loaded."
                )

            try:
                state_dict = torch.load(pretrained_pth, map_location="cpu")

                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                controller.model.load_state_dict(state_dict)
                print(f"[INFO] Loaded pretrained CNN from: {pretrained_pth}")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load pretrained CNN weights from '{pretrained_pth}': {e}"
                )

        else:
            raise RuntimeError(
                "Could not load pretrained CNN weights from the provided files. "
                "Stage 1b cannot fall back to random CNN weights."
            )

        # Freeze model parameters and switch to evaluation mode.
        if hasattr(controller, "model"):
            controller.model.eval()

            for param in controller.model.parameters():
                param.requires_grad_(False)

            print("[INFO] CNN frozen; no gradient updates will be performed.")

        logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)

        pop = max(1, int(POP))
        eval_budget = max(1, int(EVAL_BUDGET))
        max_generations = max(1, math.ceil(eval_budget / pop))

        population = [rand_bp_vec(rng) for _ in range(pop)]
        fitnesses = [None] * pop

        trial = 0
        best_so_far = -1e12

        print(f"\n{'=' * 60}")
        print("Stage 1b: Pretrained frozen CNN + evolved edge preprocessing")
        print(f"{'=' * 60}")
        print(f"Population: {pop}")
        print(f"Generations: {max_generations}")
        print(f"Eval budget: {eval_budget}")
        print(f"K-repeats: {K_REPEATS}")
        print(f"Max steps: {MAX_STEPS}")
        print(f"Seed: {SEED}")
        print(f"Run ID: {RUN_ID}")
        print(
            "Pretrained source: "
            f"pth={pretrained_pth or 'None'} | "
            f"npy={pretrained_npy or 'None'} | "
            f"json={pretrained_json or 'None'}"
        )
        print(f"{'=' * 60}\n")

        # Initial population evaluation.
        print("[INFO] Initial population evaluation")

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

            # Elitist survivor selection.
            all_individuals = list(zip(population, fitnesses)) + new_offspring
            all_individuals = [
                (genome_vec, fitness)
                for genome_vec, fitness in all_individuals
                if fitness is not None
            ]

            all_individuals.sort(key=lambda item: item[1], reverse=True)
            all_individuals = all_individuals[:pop]

            population = [genome_vec for genome_vec, _ in all_individuals]
            fitnesses = [fitness for _, fitness in all_individuals]

            print(
                f"[GEN {gen}] best={fitnesses[0]:.2f} "
                f"avg={float(np.mean(fitnesses)):.2f}"
            )

            gen += 1

        print(f"\n{'=' * 60}")
        print(f"[DONE] trials={trial}/{eval_budget}, pop={pop}, gens={gen - 1}")
        print(f"[DONE] best_so_far={best_so_far:.2f}")
        print(f"[DONE] CSV: {CSV_PATH}")
        print(f"{'=' * 60}")

    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Stage 1b: pretrained frozen CNN with evolved edge-preprocessing "
            "parameters"
        )
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained CNN weights as a PyTorch state_dict (.pth/.pt).",
    )

    parser.add_argument(
        "--pretrained_npy",
        type=str,
        default=None,
        help="Path to flat CNN weights (.npy/.npz) exported from the hybrid GA.",
    )

    parser.add_argument(
        "--pretrained_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing cnn_weights, controller_params, "
            "and/or behavior_params."
        ),
    )

    args = parser.parse_args()

    # Store CLI arguments in environment variables so main() uses one unified
    # configuration path for both CLI and shell-based execution.
    if args.pretrained:
        os.environ["PRETRAINED_CNN_PATH"] = args.pretrained

    if args.pretrained_npy:
        os.environ["PRETRAINED_CNN_NPY"] = args.pretrained_npy

    if args.pretrained_json:
        os.environ["PRETRAINED_CNN_JSON"] = args.pretrained_json

    main()
