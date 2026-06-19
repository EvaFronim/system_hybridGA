#!/usr/bin/env python3
"""
Stage 4a: Stabilized Hybrid GA — full CNN weights + behavioral parameters.

Goal:
    Use the same minimal hybrid setup as Stage 3, where each individual contains:
        1. a full flattened CNN weight vector
        2. an 8-dimensional behavioral-parameter vector

    Stage 4a adds GA-level stabilization mechanisms. It does not change the
    controller architecture, the simulator environment, or the reward definition.

Genome:
    - CNN weights:
        The CNN population is initialized from pretrained CNN weights plus small Gaussian noise.

    - 8 behavioral parameters:
        * lane_kp_off
        * lane_kp_head
        * lane_kp_curv
        * curv_thr_slow
        * lane_conf_thr
        * canny_scale
        * hough_threshold
        * diag_roi_row_start

Evaluation:
    For each individual, a controller is built using that individual's CNN
    weights. The behavioral parameters are converted to a dictionary and passed
    explicitly to evaluate_with_frozen_cnn(...).

    During each rollout, the controller weights remain fixed: no backpropagation,
    gradient descent, or online weight update is performed. The rollout return
    is the sum of simulator rewards collected during one episode.

Fitness:
    Each individual is evaluated over K repeats. Its fitness is computed as a
    trimmed mean of the K rollout returns using trim_ratio=0.10, making the
    fitness estimate more robust to extreme rollout outcomes.

GA operators:
    - CNN weights:
        * uniform crossover
        * sparse Gaussian mutation

    - behavioral parameters:
        * BLX-alpha crossover
        * bounded Gaussian mutation

Stage 4a stabilization:
    - 1-individual Hall-of-Fame:
        The best individual found so far is preserved and reinserted into the
        population if survivor selection would otherwise remove it.

    - trimmed-mean fitness aggregation:
        The K repeated rollout returns of each individual are aggregated with a
        trimmed mean instead of a plain arithmetic mean.

    - trimmed-mean generation statistic:
        The per-generation population summary is also reported with a trimmed
        mean to reduce the effect of extreme fitness values in the printed log.

Important distinction:
    The Hall-of-Fame affects survivor preservation. The trimmed mean affects the
    fitness assigned to each individual and is also used as a robust reporting
    statistic for generation-level summaries.

Purpose:
    This stage tests whether simple GA-level stabilization improves the minimal
    hybrid GA from Stage 3 without adding EMA, adaptive mutation, curriculum
    learning, reinforcement learning updates, or gradient-based CNN training.
"""

import os
import math
import argparse
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
    evaluate_with_frozen_cnn,
    _load_best_genome_from_files,
    HybridIndividual,
    mutate_cnn_vec,
    crossover_uniform_cnn,
    build_controller_from_individual,
    trimmed_mean,
)


SPACE_NAME = "hybrid_cnn_bp_hof_trimmed"
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "logs/runs_ga_stage4a_hybrid_hof_trimmed/trials.csv"
)




def evaluate_individual(env,
                        ind: HybridIndividual,
                        base_controller_params: Dict,
                        trial_idx: int) -> float:
    """
    Fitness = trimmed mean reward over K repeats.
    """
    bp_dict = bp_vec_to_dict(ind.bp_vec)
    fits: List[float] = []

    for r in range(K_REPEATS):
        s = SEED + int(RUN_ID) * 100000 + (trial_idx + 1) * 100 + r
        set_all_seeds(s)

        controller = build_controller_from_individual(ind, base_controller_params)
        if hasattr(controller, "reset_turn_stabilizer"):
            controller.reset_turn_stabilizer()

        fit = float(evaluate_with_frozen_cnn(env, controller, bp_dict,
                                             max_steps=MAX_STEPS, seed=s))
        fits.append(fit)

    return trimmed_mean(fits, trim_ratio=0.10)


def individual_log_payload(ind: HybridIndividual) -> Dict:
    return {
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_npy", type=str, default=None,
        help="(Optional) Path to pretrained CNN weights .npy (from previous hybrid run)"
    )

    parser.add_argument("--pretrained_json", type=str, required=True, help="Path to pretrained genome .json.")
    parser.add_argument("--sim_host", type=str, default=ENV_CONF["host"])
    parser.add_argument("--sim_port", type=int, default=ENV_CONF["port"])
    args = parser.parse_args()
    run_id_int = int(RUN_ID)
    effective_seed = int(SEED) + run_id_int

    set_all_seeds(effective_seed)

    if torch is not None:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    rng = np.random.default_rng(effective_seed)

    ENV_CONF["host"] = args.sim_host
    ENV_CONF["port"] = int(args.sim_port)

    genome = _load_best_genome_from_files(args.pretrained_npy, args.pretrained_json)
    if genome is None:
        raise RuntimeError("Could not load genome from npy/json – need at least cnn_weights")
    base_controller_params = dict(getattr(genome, "controller_params", {}) or {})

    cnn_weights = np.asarray(genome.cnn_weights, dtype=np.float32).ravel()
    cnn_dim = cnn_weights.size
    print(f"Loaded pretrained CNN weights: {cnn_dim} params")

    env = try_make_env(
        PRIMARY_ENV_ID,
        fallback_ids=FALLBACK_ENV_IDS,
        conf=ENV_CONF,
    )
    if env is None:
        raise RuntimeError("Cannot create Donkey env. Check simulator host/port.")

    try:
        _run_stage4a(env, base_controller_params, cnn_weights, rng, effective_seed)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_stage4a(env, base_controller_params, cnn_weights, rng, effective_seed):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)

    print("\n============================================================")
    print("Stage 4a: Stabilized Hybrid GA (Hall-of-Fame + trimmed mean)")
    print("============================================================")
    print(f"Population: {POP}")
    print(f"Eval budget: {EVAL_BUDGET}")
    print(f"K-repeats: {K_REPEATS}")
    print(f"MAX_STEPS: {MAX_STEPS}")
    print(f"CSV: {CSV_PATH}")
    print("============================================================\n")

    pop = max(1, int(POP))
    max_generations = max(1, math.ceil(EVAL_BUDGET / pop))

    population: List[HybridIndividual] = []
    for _ in range(pop):
        cnn_init = cnn_weights + rng.normal(0.0, 0.01, size=cnn_weights.shape)
        bp_init = rand_bp_vec(rng)
        population.append(HybridIndividual(cnn_init, bp_init))

    fitnesses: List[Optional[float]] = [None] * pop

    trial = 0
    best_so_far = -1e12
    global_best = None
    global_best_fit = -1e12

    print("[INFO] Initial evaluation (GEN 0)")
    for i in range(pop):
        if trial >= EVAL_BUDGET:
            break
        fit = evaluate_individual(env, population[i], base_controller_params, trial_idx=trial)
        fitnesses[i] = fit
        trial += 1
        params = individual_log_payload(population[i])
        best_so_far = logger(trial, RUN_ID, params, fit)
        print(f"[GEN 0] ind={i+1}/{pop} fit={fit:.2f} best={best_so_far:.2f}")
    if any(f is not None for f in fitnesses):
        best_idx = int(np.nanargmax([f if f is not None else -1e12 for f in fitnesses]))
        global_best = population[best_idx].copy()
        global_best_fit = float(fitnesses[best_idx])


    gen = 1
    while trial < EVAL_BUDGET and gen <= max_generations:
        print(f"\n[INFO] Generation {gen}/{max_generations}")

        def tournament() -> HybridIndividual:
            idxs = rng.choice(len(population), size=min(3, len(population)), replace=False)
            best_idx = max(idxs, key=lambda j: fitnesses[j])
            return population[best_idx]

        new_offspring: List[Tuple[HybridIndividual, float]] = []

        while len(new_offspring) < pop and trial < EVAL_BUDGET:
            p1 = tournament()
            p2 = tournament()

            # --- Crossover ---
            child_cnn = crossover_uniform_cnn(p1.cnn_vec, p2.cnn_vec, rng)
            child_bp = crossover_blend_bp_vec(p1.bp_vec, p2.bp_vec, rng, alpha=0.2)

            # --- Mutation ---
            child_cnn = mutate_cnn_vec(child_cnn, rng, sigma=0.01, p=0.05)
            child_bp = mutate_bp_vec(child_bp, rng, sigma_scale=0.10, p=0.25)

            child = HybridIndividual(child_cnn, child_bp)

            # --- Evaluate ---
            fit = evaluate_individual(env, child, base_controller_params, trial_idx=trial)
            fitnesses_str = f"{fit:.2f}"
            print(f"[DEBUG] trial={trial+1} fit={fitnesses_str}")

            trial += 1
            params = individual_log_payload(child)
            best_so_far = logger(trial, RUN_ID, params, fit)
            new_offspring.append((child, fit))

        all_ind: List[Tuple[HybridIndividual, float]] = [
            (population[i], fitnesses[i])
            for i in range(len(population))
            if fitnesses[i] is not None
        ] + new_offspring

        all_ind = [(ind, fit) for (ind, fit) in all_ind if fit is not None and np.isfinite(fit)]
        if len(all_ind) == 0:
            print("[WARN] Empty population after selection; stopping.")
            break

        all_ind.sort(key=lambda x: x[1], reverse=True)
        all_ind = all_ind[:pop]
        population = [ind for (ind, _) in all_ind]
        fitnesses = [fit for (_, fit) in all_ind]

        if fitnesses[0] > global_best_fit:
            global_best_fit = float(fitnesses[0])
            global_best = population[0].copy()

        if global_best is not None:
            in_pop = any(
                np.array_equal(ind.cnn_vec, global_best.cnn_vec)
                and np.array_equal(ind.bp_vec, global_best.bp_vec)
                for ind in population
            )
            if not in_pop:
                population[-1] = global_best.copy()
                fitnesses[-1] = global_best_fit
                combined = list(zip(population, fitnesses))
                combined.sort(key=lambda x: x[1], reverse=True)
                population = [ind for ind, _ in combined]
                fitnesses = [fit for _, fit in combined]

        valid_fits = [float(f) for f in fitnesses if f is not None and np.isfinite(f)]
        avg_trimmed = trimmed_mean(valid_fits, trim_ratio=0.10)

        print(f"[GEN {gen}] best={fitnesses[0]:.2f} avg_trimmed={avg_trimmed:.2f}")
        gen += 1

    print("\n" + "=" * 60)
    print(f"[DONE] trials={trial}/{EVAL_BUDGET}, pop={pop}, gens={gen-1}")
    print(f"[DONE] best_so_far={best_so_far:.2f}")
    print(f"[DONE] CSV: {CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
