#!/usr/bin/env python3
"""
Stage 4b: Adaptive Hybrid GA — full CNN weights + behavioral parameters.

Goal:
    Use the same stabilized hybrid setup as Stage 4a, where each individual
    contains:
        1. a full flattened CNN weight vector
        2. an 8-dimensional behavioral-parameter vector

    Stage 4b keeps the Stage 4a stabilization mechanisms and adds EMA-based
    adaptive mutation. It does not change the controller architecture, the
    simulator environment, or the reward definition.

Genome:
    - CNN weights:
        Full flattened CNN weight vector, initialized from pretrained weights
        plus small Gaussian noise.

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
    trimmed mean of the K rollout returns using trim_ratio=0.10, as in Stage 4a.

GA operators:
    - CNN weights:
        * uniform crossover
        * sparse Gaussian mutation

    - behavioral parameters:
        * BLX-alpha crossover
        * bounded Gaussian mutation

Stage 4b mechanisms:
    - 1-individual Hall-of-Fame:
        The best individual found so far is preserved and reinserted into the
        population if survivor selection would otherwise remove it.

    - trimmed-mean fitness aggregation:
        The K repeated rollout returns of each individual are aggregated with a
        trimmed mean instead of a plain arithmetic mean.

    - trimmed-mean generation statistic:
        The per-generation population summary is also reported with a trimmed
        mean to reduce the effect of extreme fitness values in the printed log.

    - EMA-based adaptive mutation:
    The CNN mutation rate starts from 0.05 with mutation strength 0.01,
    matching the Stage 4a CNN mutation configuration. The behavioral-parameter
    mutation rate starts from 0.25 with mutation strength 0.10, also matching
    Stage 4a. Both rates are adapted using EMA decay when progress is observed
    and are temporarily boosted after repeated stagnation. Separate bounds are
    used for CNN weights and behavioral parameters due to their different
    dimensionality and sensitivity.

Purpose:
    This stage tests whether adding EMA-based adaptive mutation on top of the
    Stage 4a stabilization mechanisms improves the exploitation-exploration balance.
"""

import os
import math
import json
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


SPACE_NAME = "hybrid_cnn_bp_hof_trimmed_adaptive"
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "logs/runs_ga_stage4b_hybrid_adaptive/trials.csv"
)



def save_hybrid_genome(path: str,
                       ind: "HybridIndividual",
                       base_controller_params: dict):
    """
    Save a hybrid GA individual to a JSON file.
    The saved policy contains:
     -the full flattened CNN weight vector
     -the decoded behavioral parameters
     -controller parameters loaded from the pretrained genome
    This file is intended for later neutral evaluation or rollout replay, where
    the best evolved policy must be reconstructed exactly.
    Unlike the CSV logger, which stores only summary statistics of the CNN
    weights, this function stores the full CNN genome.
    """
    payload = {
        "cnn_weights": ind.cnn_vec.tolist(),
        "behavior_params": bp_vec_to_dict(ind.bp_vec),
        "controller_params": base_controller_params or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)



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


def _ema(prev: float, target: float, alpha: float) -> float:
    """
    Exponential Moving Average update.
    Used to smooth the adaptive mutation-rate updates for both parts of the
    hybrid genome:
        - CNN mutation rate
        - behavioral-parameter mutation rate
    prev:
        Previous EMA value.
    target:
        Target value toward which the EMA should move.
    alpha:
        Smoothing factor in [0, 1].
        Larger values make the EMA follow the target faster.
    """
    return (1.0 - alpha) * prev + alpha * target

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
    help="(Optional) Path to pretrained CNN weights .npy (from previous hybrid run)")
    parser.add_argument("--pretrained_json", type=str, required=True,
                        help="Path to pretrained genome .json (for controller_params meta)")
    parser.add_argument("--sim_host", type=str, default=ENV_CONF["host"])
    parser.add_argument("--sim_port", type=int, default=ENV_CONF["port"])
    args = parser.parse_args()
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
        _run_stage4b(env, base_controller_params, cnn_weights, rng, effective_seed)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_stage4b(env, base_controller_params, cnn_weights, rng, effective_seed):
    base_cnn_mutation_rate = 0.05
    base_cnn_mutation_strength = 0.01

    base_bp_mutation_rate = 0.25
    base_bp_mutation_strength = 0.10

    mrate_decay_per_gen = 0.995
    ema_alpha = 0.2

    # CNN bounds
    cnn_mrate_min = 0.025
    cnn_mrate_max = 0.15
    cnn_mstr_min = 0.005
    cnn_mstr_max = 0.03

    # BP bounds
    bp_mrate_min = 0.10
    bp_mrate_max = 0.50
    bp_mstr_min = 0.05
    bp_mstr_max = 0.20

    min_improvement = 0.009

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)

    print("\n============================================================")
    print("Stage 4b: Adaptive Hybrid GA")
    print("CNN + Behavior Params + HoF + Trimmed Mean + Adaptive Mutation")
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

        save_hybrid_genome(
            path=os.path.join(
                "saved_policies",
                "stage4b",
                f"run{RUN_ID}_gen0_fit{global_best_fit:.2f}.json",
            ),
            ind=global_best,
            base_controller_params=base_controller_params,
        )


    ema_cnn_mutation_rate = base_cnn_mutation_rate   
    ema_bp_mutation_rate = base_bp_mutation_rate
    stagnation_counter = 0                  
    prev_gen_best = float(global_best_fit)   
   

    gen = 1
    while trial < EVAL_BUDGET and gen <= max_generations:
        print(f"\n[INFO] Generation {gen}/{max_generations}")

        # --- Snapshot mutation values for this generation ---
        curr_cnn_mrate = float(np.clip(base_cnn_mutation_rate,cnn_mrate_min,cnn_mrate_max))
        curr_cnn_mstr = float(np.clip(base_cnn_mutation_strength,cnn_mstr_min,cnn_mstr_max))

        curr_bp_mrate = float(np.clip(base_bp_mutation_rate,bp_mrate_min,bp_mrate_max))
        curr_bp_mstr = float(np.clip(base_bp_mutation_strength,bp_mstr_min,bp_mstr_max))


        def tournament() -> HybridIndividual:
            idxs = rng.choice(len(population), size=min(3, len(population)), replace=False)
            best_idx = max(idxs, key=lambda j: fitnesses[j])
            return population[best_idx]

        new_offspring: List[Tuple[HybridIndividual, float]] = []

        while len(new_offspring) < pop and trial < EVAL_BUDGET:
            p1 = tournament()
            p2 = tournament()

            child_cnn = crossover_uniform_cnn(p1.cnn_vec, p2.cnn_vec, rng)
            child_bp = crossover_blend_bp_vec(p1.bp_vec, p2.bp_vec, rng, alpha=0.2)

            child_cnn = mutate_cnn_vec(child_cnn, rng, sigma=curr_cnn_mstr, p=curr_cnn_mrate)
            child_bp  = mutate_bp_vec(child_bp, rng, sigma_scale=curr_bp_mstr,p=curr_bp_mrate) 


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

        gen_best = float(fitnesses[0])


        # --- Global-best Hall-of-Fame ---
        if fitnesses[0] > global_best_fit:
            global_best_fit = float(fitnesses[0])
            global_best = population[0].copy()

            save_hybrid_genome(
                path=os.path.join(
                    "saved_policies",
                    "stage4b",
                    f"run{RUN_ID}_gen{gen}_fit{global_best_fit:.2f}.json",
                ),
                ind=global_best,
                base_controller_params=base_controller_params,
            )

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

        # --- Stage 4b: adaptive mutation with EMA + stagnation ---
        improvement = gen_best - prev_gen_best

        if improvement < min_improvement:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        prev_gen_best = gen_best

        if stagnation_counter == 0:
            # Progress: decay mutation rates smoothly
            cnn_target = base_cnn_mutation_rate * mrate_decay_per_gen
            bp_target = base_bp_mutation_rate * mrate_decay_per_gen

            ema_cnn_mutation_rate = _ema(ema_cnn_mutation_rate,cnn_target,ema_alpha)

            ema_bp_mutation_rate = _ema(ema_bp_mutation_rate,bp_target,ema_alpha)

            base_cnn_mutation_rate = float(np.clip(ema_cnn_mutation_rate,cnn_mrate_min,cnn_mrate_max))
            base_bp_mutation_rate = float(np.clip(ema_bp_mutation_rate,bp_mrate_min,bp_mrate_max))

        else:
            # Stagnation: boost mutation rates
            if stagnation_counter > 2:
                boost = 1.0 + 0.25 * (stagnation_counter - 2)

                base_cnn_mutation_rate = float(np.clip(base_cnn_mutation_rate * boost,cnn_mrate_min,cnn_mrate_max))
                base_bp_mutation_rate = float(np.clip(base_bp_mutation_rate * boost,bp_mrate_min,bp_mrate_max))

                ema_cnn_mutation_rate = base_cnn_mutation_rate
                ema_bp_mutation_rate = base_bp_mutation_rate

        # Couple mutation strength to mutation rate
        base_cnn_mutation_strength = float(np.clip(0.01 * (base_cnn_mutation_rate / 0.05),cnn_mstr_min,cnn_mstr_max))

        base_bp_mutation_strength = float(np.clip(0.10 * (base_bp_mutation_rate / 0.25),bp_mstr_min,bp_mstr_max))

        gen += 1


    print("\n" + "=" * 60)
    print(f"[DONE] trials={trial}/{EVAL_BUDGET}, pop={pop}, gens={gen-1}")
    print(f"[DONE] best_so_far={best_so_far:.2f}")
    print(f"[DONE] CSV: {CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
