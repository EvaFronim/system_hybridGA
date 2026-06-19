# ============================================================================
# Stage 2b: Co-evolution of behavioral parameters and CNN adapters.
#
# Genome:
#   - 8 behavioral parameters:
#       * lane_kp_off, lane_kp_head, lane_kp_curv, curv_thr_slow
#       * lane_conf_thr, canny_scale, hough_threshold, diag_roi_row_start
#
#   - 4*G CNN adapter parameters:
#       * G grouped scales for the last convolutional layer
#       * G grouped shifts for the last convolutional layer
#       * G grouped scales for the last hidden linear/head layer
#       * G grouped shifts for the last hidden linear/head layer
#
# With G=8, the adapter genome has 32 parameters:
#   8 conv scales + 8 conv shifts + 8 head scales + 8 head shifts.
#
# Total genome size:
#   8 behavioral parameters + 32 adapter parameters = 40 parameters.
#
# The pretrained CNN weights remain frozen. The GA evolves only:
#   1. behavioral/preprocessing parameters
#   2. lightweight adapter scale/shift parameters
#
# Fitness:
#   Mean rollout return over K repeats, where each rollout return is the sum of
#   environment rewards accumulated during the episode.
#
# Purpose:
#   This stage tests whether co-evolving CNN adapters and behavioral parameters
#   improves performance compared with evolving only one component family.
#
# Implementation note:
#   The adapter hooks modify activations at runtime. They do not update the
#   pretrained CNN weights through gradient descent.
#
# Behavioral-parameter note:
#   The full 8-parameter behavioral schema is co-evolved for consistency with
#   the rest of the study. In the current end-to-end controller path, the four
#   vision-related parameters affect preprocessing directly:
#       * lane_conf_thr
#       * canny_scale
#       * hough_threshold
#       * diag_roi_row_start
#
#   The four lane/curvature control parameters are logged and evolved for schema
#   consistency, but do not directly enter the current controller.predict(edge_img)
#   control path:
#       * lane_kp_off
#       * lane_kp_head
#       * lane_kp_curv
#       * curv_thr_slow
#
# Distinction from Stage 2a:
#   Stage 2a adapts the final controller outputs directly, typically steering
#   and throttle. Stage 2b applies grouped adapters to the last convolutional
#   layer and to the last hidden linear/head layer before the final 2-output
#   layer.
# ============================================================================

from __future__ import annotations
import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
from scripts.main_ga import try_make_env

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

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
    clamp_bp_vec,
    bp_vec_to_dict,
    evaluate_with_frozen_cnn,
    _load_best_genome_from_files,
    controller_cfg_genome,
)


SPACE_NAME  = "CoEvo-BP+Adapters"

LOG_DIR  = os.path.join("logs", "runs_ga_stage2_2_coevo")
CSV_PATH = os.path.join(LOG_DIR, "trials.csv")
os.makedirs(LOG_DIR, exist_ok=True)

BP_DIM = len(BP_BOUNDS)


# Adapters (grouped scale/shift)
ADAPTER_GROUPS = int(os.environ.get("ADAPTER_GROUPS", "8"))  # G
ADAPTER_BOUNDS_PER_GROUP = {
    "conv_scale": (0.80, 1.20),
    "conv_shift": (-0.20, 0.20),
    "head_scale": (0.80, 1.20),
    "head_shift": (-0.20, 0.20),
}
ADAPTER_DIM = ADAPTER_GROUPS * 4
GENOME_DIM  = BP_DIM + ADAPTER_DIM


class _AdapterMapper:
    def __init__(self, groups: int, conv_channels: int, head_dim: int):
        assert groups >= 1
        self.G = groups; self.C = conv_channels; self.H = head_dim
        self.conv_slices = self._slices(self.C, self.G)
        self.head_slices = self._slices(self.H, self.G)

    @staticmethod
    def _slices(n, g):
        base, rem = n // g, n % g
        sizes = [base + (1 if i < rem else 0) for i in range(g)]
        out = []; st = 0
        for sz in sizes:
            out.append((st, st+sz)); st += sz
        return out

    def split(self, vec: np.ndarray) -> Dict[str, np.ndarray]:
        g = self.G
        assert vec.shape[0] == 4*g
        c_sc = vec[0*g:1*g]; c_sh = vec[1*g:2*g]
        h_sc = vec[2*g:3*g]; h_sh = vec[3*g:4*g]
        conv_scales = np.ones((self.C,), dtype=np.float32)
        conv_shifts = np.zeros((self.C,), dtype=np.float32)
        head_scales = np.ones((self.H,), dtype=np.float32)
        head_shifts = np.zeros((self.H,), dtype=np.float32)
        for gi,(a,b) in enumerate(self.conv_slices):
            if b>a: conv_scales[a:b] = c_sc[gi]; conv_shifts[a:b] = c_sh[gi]
        for gi,(a,b) in enumerate(self.head_slices):
            if b>a: head_scales[a:b] = h_sc[gi]; head_shifts[a:b] = h_sh[gi]
        return {"conv_scales":conv_scales,"conv_shifts":conv_shifts,
                "head_scales":head_scales,"head_shifts":head_shifts}

def _rand_adapters(rng) -> np.ndarray:
    g = ADAPTER_GROUPS
    vec = np.empty((4*g,), dtype=np.float32)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["conv_scale"]; vec[0*g:1*g] = rng.uniform(lo, hi, size=g)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["conv_shift"]; vec[1*g:2*g] = rng.uniform(lo, hi, size=g)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["head_scale"]; vec[2*g:3*g] = rng.uniform(lo, hi, size=g)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["head_shift"]; vec[3*g:4*g] = rng.uniform(lo, hi, size=g)
    return vec

def _clamp_adapters(v: np.ndarray) -> np.ndarray:
    g = ADAPTER_GROUPS
    out = v.copy()
    def clamp_range(seg, bounds):
        lo, hi = bounds; seg[:] = np.clip(seg, lo, hi)
    clamp_range(out[0*g:1*g], ADAPTER_BOUNDS_PER_GROUP["conv_scale"])
    clamp_range(out[1*g:2*g], ADAPTER_BOUNDS_PER_GROUP["conv_shift"])
    clamp_range(out[2*g:3*g], ADAPTER_BOUNDS_PER_GROUP["head_scale"])
    clamp_range(out[3*g:4*g], ADAPTER_BOUNDS_PER_GROUP["head_shift"])
    return out

def _rand_genome(rng, bp_seed: np.ndarray | None = None) -> np.ndarray:
    bp = clamp_bp_vec(bp_seed) if bp_seed is not None else rand_bp_vec(rng)
    ad = _rand_adapters(rng)
    return np.concatenate([bp, ad]).astype(np.float32)

def _split_genome(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return g[:BP_DIM], g[BP_DIM:]


def mutate(
    genome: np.ndarray,
    rng,
    bp_sigma_scale=0.10,
    bp_p=0.25,
    ad_sigma=0.05,
    ad_p=0.25,
) -> np.ndarray:
    """
   Mutation Operator.
    
    Applies Gaussian noise with distinct intensities for:
    - Behavioral parameters (coarser tuning).
    - Adapter weights (finer neural tuning).
    """
    
    y = genome.copy()
    # BP
    for i,(lo,hi) in enumerate(BP_BOUNDS):
        if rng.random() < bp_p:
            sigma = bp_sigma_scale * (hi - lo)
            y[i] += rng.normal(0.0, sigma)
            y[i] = min(hi, max(lo, float(y[i])))
    # Adapters
    g = ADAPTER_GROUPS; off = BP_DIM
    for i in range(4*g):
        if rng.random() < ad_p:
            y[off+i] += rng.normal(0.0, ad_sigma)
    y[BP_DIM:] = _clamp_adapters(y[BP_DIM:])
    return y

def crossover_blend(a: np.ndarray, b: np.ndarray, rng, alpha=0.2) -> np.ndarray:
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    span = hi - lo
    child = rng.uniform(lo - alpha*span, hi + alpha*span)
    child[:BP_DIM] = clamp_bp_vec(child[:BP_DIM])
    child[BP_DIM:] = _clamp_adapters(child[BP_DIM:])
    return child

# Hooks: apply adapters to conv/head
def _find_last_conv_and_head(model: nn.Module):
    last_conv = None
    linear_layers = []

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
        if isinstance(m, nn.Linear):
            linear_layers.append(m)

    if len(linear_layers) >= 2:
        # Use the last hidden linear layer before the final 2-output layer.
        head = linear_layers[-2]
    elif len(linear_layers) == 1:
        # Fallback: only one linear layer exists.
        head = linear_layers[-1]
    else:
        head = None

    return last_conv, head

def _detect_adapter_dims(controller):
    C = 64
    H = 32
    if hasattr(controller, "model") and nn is not None:
        last_conv, last_linear = _find_last_conv_and_head(controller.model)
        if last_conv is not None and hasattr(last_conv, "out_channels"):
            C = int(last_conv.out_channels)
        if last_linear is not None and hasattr(last_linear, "out_features"):
            H = int(last_linear.out_features)
    return C, H


def apply_adapters(controller, expanded):
    if torch is None or not hasattr(controller, "model"):
        return lambda: None
    model = controller.model
    last_conv, head = _find_last_conv_and_head(model)
    hooks = []
    if last_conv is not None:
        cs = torch.from_numpy(expanded["conv_scales"].copy()).view(1, -1, 1, 1).float()
        ch = torch.from_numpy(expanded["conv_shifts"].copy()).view(1, -1, 1, 1).float()
        def _hook_conv(_m, _inp, out):
            return out * cs.to(out.device) + ch.to(out.device)
        hooks.append(last_conv.register_forward_hook(lambda m,i,o: _hook_conv(m,i,o)))
    if head is not None:
        hs = torch.from_numpy(expanded["head_scales"].copy()).view(1, -1).float()
        hh = torch.from_numpy(expanded["head_shifts"].copy()).view(1, -1).float()
        def _hook_head(_m, _inp, out):
            return out * hs.to(out.device) + hh.to(out.device)
        hooks.append(head.register_forward_hook(lambda m,i,o: _hook_head(m,i,o)))
    def _remove():
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
    return _remove


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained_npy',   default=None)
    ap.add_argument('--pretrained_json',  default=None)
    ap.add_argument('--bp_init_json',     default=None, help='Optional BP seed (mutable).')
    
    # NOTE: defaults below are specific to the original development machine's
    # WSL2 network. Override with --sim_host/--sim_port (or SIM_HOST/SIM_PORT
    # env vars, which ENV_CONF already reads) when running elsewhere.
    ap.add_argument('--sim_host',         default=os.environ.get('SIM_HOST', '127.0.0.1'))
    ap.add_argument('--sim_port', type=int, default=9091)
    args = ap.parse_args()

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

    # Env
    if args.sim_host:
        ENV_CONF["host"] = args.sim_host

    if args.sim_port:
        ENV_CONF["port"] = int(args.sim_port)

    env = try_make_env(PRIMARY_ENV_ID, fallback_ids=FALLBACK_ENV_IDS, conf=ENV_CONF)
    if env is None:
        raise RuntimeError("Cannot open Donkey env. Is the simulator running?")

    try:
        _run_coevolution(env, args, run_id_int, effective_seed, rng)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_coevolution(env, args, run_id_int, effective_seed, rng):
    # Pretrained controller (frozen CNN) 
    genome = _load_best_genome_from_files(args.pretrained_npy, args.pretrained_json)
    if genome is None:
        raise RuntimeError(
            "Stage 2b requires pretrained CNN weights. "
            "Provide --pretrained_npy and/or --pretrained_json."
        )

    load_w = True

    init_errors: List[str] = []
    controller = None
    for sig in [
        lambda: Controller(genome=genome, config=ControllerConfig(), load_weights=load_w),
        lambda: Controller(genome=genome, load_weights=load_w),
        lambda: controller_cfg_genome(genome),
    ]:
        try:
            controller = sig(); break
        except Exception as e:
            init_errors.append(str(e)); controller = None
    if controller is None:
        raise RuntimeError("Controller init failed: " + " | ".join(init_errors))
    if torch is not None and hasattr(controller, "model"):
        controller.model.eval()
        for p in controller.model.parameters():
            p.requires_grad_(False)
        print("CNN frozen (no gradient updates)")

    # BP seed (mutable)
    bp_seed = None
    if args.bp_init_json and os.path.isfile(args.bp_init_json):
        try:
            with open(args.bp_init_json, 'r') as f:
                bp_json = json.load(f)
            bp_seed = np.array([float(bp_json[k]) for k in PARAM_KEYS], dtype=np.float32)
            print("BP seed loaded (mutable).")
        except Exception as e:
            print(f"[WARN] Could not read bp_init_json: {e}")

    # Mapper
    g = ADAPTER_GROUPS
    detC, detH = _detect_adapter_dims(controller)
    conv_C = int(os.environ.get("ADAPTER_LAST_CONV_C", detC))
    head_H = int(os.environ.get("ADAPTER_HEAD_DIM",    detH))
    print(f"[Adapters] Using last_conv_C={conv_C}, head_dim={head_H} (detected {detC}/{detH})")
    mapper = _AdapterMapper(g, conv_channels=conv_C, head_dim=head_H)

    # Logger & banner
    logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)
    print("\n============================================================")
    print("Stage 2.2: Co-evolution (BP + CNN Adapters, frozen CNN)")
    print("============================================================")
    print(f"Population: {POP}")
    print(f"Eval budget: {EVAL_BUDGET}")
    print(f"K-repeats: {K_REPEATS}")
    print(f"Adapter groups: {g}")
    print(f"CSV: {CSV_PATH}")
    print("============================================================\n")

    pop = max(1, int(POP))
    max_generations = max(1, math.ceil(EVAL_BUDGET / pop))
    population = [_rand_genome(rng, bp_seed=bp_seed) for _ in range(pop)]
    fitnesses: List[float|None] = [None]*pop

    trial = 0
    best_so_far = -1e12

    def eval_one(genome_vec: np.ndarray, trial_idx: int) -> float:
        bp_vec, ad_vec = _split_genome(genome_vec)
        params = bp_vec_to_dict(bp_vec)
        expanded = mapper.split(ad_vec)
        cleanup = apply_adapters(controller, expanded)  # hook adapters
        try:
            fits = []
            for r in range(K_REPEATS):
                s = SEED + int(RUN_ID)*100000 + (trial_idx + 1)*100 + r
                set_all_seeds(s)
                fits.append(evaluate_with_frozen_cnn(env, controller, params, max_steps=MAX_STEPS, seed=s))
            fit = float(np.mean(fits))
        finally:
            cleanup()  # remove hooks
        payload = {
            "bp": params,
            "adapters_groups": {
                "groups": g,
                "conv_scale": ad_vec[0*g:1*g].tolist(),
                "conv_shift": ad_vec[1*g:2*g].tolist(),
                "head_scale": ad_vec[2*g:3*g].tolist(),
                "head_shift": ad_vec[3*g:4*g].tolist(),
            }
        }
        nonlocal best_so_far
        best_so_far = logger(trial_idx+1, RUN_ID, payload, fit)
        if os.environ.get("VERBOSE", "0") == "1":
            print(f"[trial {trial_idx:03d}] run={RUN_ID} fit={fit:.3f} best={best_so_far:.3f}")
        return fit

    # Initial eval
    for i in range(pop):
        if trial >= EVAL_BUDGET: break
        fitnesses[i] = eval_one(population[i], trial_idx=trial); trial += 1

    def tournament():
        idxs = rng.choice(
            len(population),
            size=min(3, len(population)),
            replace=False,
        )
        return population[max(idxs, key=lambda j: fitnesses[j])]

    gen = 1
    while trial < EVAL_BUDGET and gen <= max_generations:
        offspring: List[Tuple[np.ndarray,float]] = []
        while len(offspring) < pop and trial < EVAL_BUDGET:
            p1, p2 = tournament(), tournament()
            child = crossover_blend(p1, p2, rng, alpha=0.2)
            child = mutate(child, rng, bp_sigma_scale=0.10, bp_p=0.25, ad_sigma=0.05, ad_p=0.25)
            f = eval_one(child, trial_idx=trial); trial += 1
            offspring.append((child, f))
        # elitist selection
        pool = list(zip(population, fitnesses)) + offspring
        pool = [(v,f) for (v,f) in pool if f is not None]
        pool.sort(key=lambda x: x[1], reverse=True)
        pool = pool[:pop]
        population = [v for (v,_) in pool]
        fitnesses  = [f for (_,f) in pool]
        gen += 1

    print(f"[CoEvo] trials={trial}/{EVAL_BUDGET}, population={pop}, generations≈{max_generations}")
    print(f"[CoEvo] CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
