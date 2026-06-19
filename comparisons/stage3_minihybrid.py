#!/usr/bin/env python3
"""
Stage 3: Minimal Hybrid GA — full CNN weights + behavioral parameters.

Goal:
    Evolve the full flattened CNN weight vector and the 8 behavioral parameters
    in the same genome, using the same simulator evaluation protocol as the
    previous stages.

Genome:
    - CNN weights:
        Full flattened weight vector loaded from a pretrained genome and then
        evolved directly with genetic operators.

    - 8 behavioral parameters:
        * lane_kp_off
        * lane_kp_head
        * lane_kp_curv
        * curv_thr_slow
        * lane_conf_thr
        * canny_scale
        * hough_threshold
        * diag_roi_row_start

Initialization:
    The initial CNN population is created by adding small Gaussian noise to the
    pretrained CNN weights. Behavioral parameters are initialized uniformly
    within the shared BP bounds.

    Note: Initialization noise is dense (applied to every CNN weight with
    sigma=0.01), whereas mutation noise is sparse (5% of weights per event
    with sigma=0.01). This produces broader initial population diversity than
    any single mutation step can introduce.

Evaluation:
    For each individual, a controller is built using that individual's CNN
    weights and behavioral parameters. The controller is evaluated in the
    simulator using the shared rollout function evaluate_with_frozen_cnn(...).

    During a rollout, the controller weights are kept fixed: no backpropagation,
    gradient descent, or online weight update is performed. The resulting
    fitness score is then used by the genetic algorithm to select, crossover,
    and mutate individuals across generations.

    Therefore, the CNN is optimized by evolutionary search, not by gradient-based
    training.

Important distinction:
    Although evaluate_with_frozen_cnn does not perform gradient updates, the CNN
    weights are not fixed in this stage. They are modified by the genetic
    algorithm between evaluations. During each rollout, the instantiated
    controller is evaluated without learning.

GA operators:
    - CNN weights:
        * uniform crossover
        * sparse Gaussian mutation

    - behavioral parameters:
        * BLX-alpha crossover
        * bounded Gaussian mutation

Purpose:
    This stage tests whether directly evolving the full CNN weight vector
    together with behavioral parameters improves performance compared with:
        * BP-only evolution
        * frozen random CNN + BP evolution
        * frozen pretrained CNN + BP evolution
        * lightweight CNN adapter evolution
        * BP + CNN adapter co-evolution

Scope:
    This is intentionally a minimal hybrid GA. It does not use curriculum
    learning, EMA, reinforcement learning updates, backpropagation, or additional
    training components.
"""

import os
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

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
    PARAM_KEYS,
    BP_BOUNDS,
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
)


SPACE_NAME = "hybrid_cnn_bp"
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "logs/runs_ga_hybrid_minimal/trials.csv"
)


def evaluate_individual(
    env,
    ind: HybridIndividual,
    base_controller_params: Dict,
    trial_idx: int,
) -> float:
    """
    Evaluate one hybrid individual over K repeated simulator rollouts.

    For each repeat, a controller is reconstructed from the individual's CNN
    weight vector, while the behavioral-parameter vector is decoded and passed
    explicitly to evaluate_with_frozen_cnn(...).

    The CNN weights remain fixed during each rollout. No backpropagation,
    gradient descent, or online update is performed.

    Fitness is computed as the arithmetic mean of the K rollout returns.

    Seeds follow the shared stage convention:
        s = SEED + RUN_ID * 100000 + (trial_idx + 1) * 100 + r
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

    return float(np.mean(fits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--pretrained_npy", type=str, default=None,
    help="(Optional) Path to pretrained CNN weights .npy (from previous hybrid run)")
    parser.add_argument("--pretrained_json", type=str, required=True,
                        help="Path to pretrained genome .json (for controller_params meta)")
    parser.add_argument("--sim_host", type=str, default=ENV_CONF["host"])
    parser.add_argument("--sim_port", type=int, default=ENV_CONF["port"])
    args = parser.parse_args()
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

    ENV_CONF["host"] = args.sim_host
    ENV_CONF["port"] = int(args.sim_port)

    genome = _load_best_genome_from_files(args.pretrained_npy, args.pretrained_json)
    if genome is None:
        raise RuntimeError("Could not load genome from npy/json – need at least cnn_weights")

    cnn_weights = np.asarray(genome.cnn_weights, dtype=np.float32).ravel()
    cnn_dim = cnn_weights.size

    base_controller_params = dict(getattr(genome, "controller_params", {}) or {})

    print(f"Loaded pretrained CNN weights from genome: {cnn_dim} params")


    env = try_make_env(
        PRIMARY_ENV_ID,
        fallback_ids=FALLBACK_ENV_IDS,
        conf=ENV_CONF,
    )
    if env is None:
        raise RuntimeError("Cannot create Donkey env. Check simulator host/port.")

    try:
        _run_stage3(env, cnn_weights, base_controller_params, rng)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_stage3(env, cnn_weights, base_controller_params, rng):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)

    print("\n============================================================")
    print("Minimal Hybrid GA: CNN + Behavior Params")
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

    print("[INFO] Initial evaluation (GEN 0)")
    for i in range(pop):
        if trial >= EVAL_BUDGET:
            break
        fit = evaluate_individual(env, population[i], base_controller_params, trial_idx=trial)
        fitnesses[i] = fit
        trial += 1
        params =  bp_vec_to_dict(population[i].bp_vec)
        best_so_far = logger(trial, RUN_ID, params, fit)
        print(f"[GEN 0] ind={i+1}/{pop} fit={fit:.2f} best={best_so_far:.2f}")

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

            # Crossover
            child_cnn = crossover_uniform_cnn(p1.cnn_vec, p2.cnn_vec, rng)
            child_bp = crossover_blend_bp_vec(p1.bp_vec, p2.bp_vec, rng, alpha=0.2)

            # Mutation
            child_cnn = mutate_cnn_vec(child_cnn, rng, sigma=0.01, p=0.05)
            child_bp = mutate_bp_vec(child_bp, rng, sigma_scale=0.10, p=0.25)

            child = HybridIndividual(child_cnn, child_bp)

            # Evaluate 
            fit = evaluate_individual(env, child, base_controller_params, trial_idx=trial)
            print(f"[DEBUG] trial={trial+1} fit={fit:.2f}")

            trial += 1
            params = bp_vec_to_dict(child.bp_vec)
            best_so_far = logger(trial, RUN_ID, params, fit)
            new_offspring.append((child, fit))

        # Elitism
        all_ind: List[Tuple[HybridIndividual, float]] = [
            (population[i], fitnesses[i]) for i in range(len(population)) if fitnesses[i] is not None
        ] + new_offspring

        all_ind.sort(key=lambda x: x[1], reverse=True)
        all_ind = all_ind[:pop]
        population = [ind for (ind, _) in all_ind]
        fitnesses = [fit for (_, fit) in all_ind]

        print(f"[GEN {gen}] best={fitnesses[0]:.2f} avg={float(np.mean(fitnesses)):.2f}")
        gen += 1

    print("\n" + "=" * 60)
    print(f"[DONE] trials={trial}/{EVAL_BUDGET}, pop={pop}, gens={gen-1}")
    print(f"[DONE] best_so_far={best_so_far:.2f}")
    print(f"[DONE] CSV: {CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
