"""
Stage 2a (stage2_evolveCNN_only.py) 
Ablation Study: Grouped Adapters with Fixed Behavioral Parameters.

This stage isolates the impact of Neural Adapters by evolving only the scale 
and shift parameters of the CNN's final layers, while keeping the 
high-level driving behavior (PID/Lane-keeping) fixed.

The goal is to answer: "Can we adapt a frozen feature extractor to a new 
environment just by shifting its activations?"
"""

from __future__ import annotations

import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

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
    set_all_seeds,
    make_logger,
    clamp_bp_dict,
    evaluate_with_frozen_cnn,
    _load_best_genome_from_files,
    controller_cfg_genome,
)


SPACE_NAME = "Adapters-only_BPfixed_matched"

LOG_DIR = os.path.join("logs", "runs_ga_stage2_evolve_cnn_only")
CSV_PATH = os.path.join(LOG_DIR, "trials.csv")
os.makedirs(LOG_DIR, exist_ok=True)


# Adapters (grouped scale/shift) 
ADAPTER_GROUPS = int(os.environ.get("ADAPTER_GROUPS", "8"))   # G
ADAPTER_BOUNDS_PER_GROUP = {
    "conv_scale": (0.80, 1.20),
    "conv_shift": (-0.20, 0.20),
    "head_scale": (0.80, 1.20),
    "head_shift": (-0.20, 0.20),
}
ADAPTER_DIM = ADAPTER_GROUPS * 4


class _AdapterMapper:
    """
    Expand grouped adapter genes to per-channel / per-unit vectors.
    
    Using a Grouped Adapter approach (G groups), this class expands the 
    genotype (32 parameters) into full-width vectors matching the CNN's 
    channel/unit counts (C and H).
    """

    def __init__(self, groups: int, conv_channels: int, head_dim: int):
        assert groups >= 1
        self.G = groups
        self.C = conv_channels
        self.H = head_dim
        self.conv_slices = self._slices(self.C, self.G)
        self.head_slices = self._slices(self.H, self.G)

    @staticmethod
    def _slices(n: int, g: int) -> List[Tuple[int, int]]:
        base, rem = n // g, n % g
        sizes = [base + (1 if i < rem else 0) for i in range(g)]
        out: List[Tuple[int, int]] = []
        st = 0
        for sz in sizes:
            out.append((st, st + sz))
            st += sz
        return out

    def split(self, vec: np.ndarray) -> Dict[str, np.ndarray]:
        g = self.G
        assert vec.shape[0] == 4 * g
        c_sc = vec[0 * g:1 * g]
        c_sh = vec[1 * g:2 * g]
        h_sc = vec[2 * g:3 * g]
        h_sh = vec[3 * g:4 * g]

        conv_scales = np.ones((self.C,), dtype=np.float32)
        conv_shifts = np.zeros((self.C,), dtype=np.float32)
        head_scales = np.ones((self.H,), dtype=np.float32)
        head_shifts = np.zeros((self.H,), dtype=np.float32)

        for gi, (a, b) in enumerate(self.conv_slices):
            if b > a:
                conv_scales[a:b] = c_sc[gi]
                conv_shifts[a:b] = c_sh[gi]
        for gi, (a, b) in enumerate(self.head_slices):
            if b > a:
                head_scales[a:b] = h_sc[gi]
                head_shifts[a:b] = h_sh[gi]

        return {
            "conv_scales": conv_scales,
            "conv_shifts": conv_shifts,
            "head_scales": head_scales,
            "head_shifts": head_shifts,
        }


def _rand_adapters(rng) -> np.ndarray:
    g = ADAPTER_GROUPS
    vec = np.empty((4 * g,), dtype=np.float32)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["conv_scale"]
    vec[0 * g:1 * g] = rng.uniform(lo, hi, size=g)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["conv_shift"]
    vec[1 * g:2 * g] = rng.uniform(lo, hi, size=g)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["head_scale"]
    vec[2 * g:3 * g] = rng.uniform(lo, hi, size=g)
    lo, hi = ADAPTER_BOUNDS_PER_GROUP["head_shift"]
    vec[3 * g:4 * g] = rng.uniform(lo, hi, size=g)
    return vec


def _clamp_adapters(v: np.ndarray) -> np.ndarray:
    g = ADAPTER_GROUPS
    out = v.copy()

    def clamp_range(seg: np.ndarray, bounds: Tuple[float, float]) -> None:
        lo, hi = bounds
        seg[:] = np.clip(seg, lo, hi)

    clamp_range(out[0 * g:1 * g], ADAPTER_BOUNDS_PER_GROUP["conv_scale"])
    clamp_range(out[1 * g:2 * g], ADAPTER_BOUNDS_PER_GROUP["conv_shift"])
    clamp_range(out[2 * g:3 * g], ADAPTER_BOUNDS_PER_GROUP["head_scale"])
    clamp_range(out[3 * g:4 * g], ADAPTER_BOUNDS_PER_GROUP["head_shift"])
    return out


def mutate_adapters(
    vec: np.ndarray,
    rng,
    sigma_scale: float = 0.10,
    p: float = 0.25,
) -> np.ndarray:
    """Per-gene Gaussian mutation with bounded clamping.

    The sigma for each gene is scaled by that gene's allowed range, so a
    fixed sigma_scale produces consistent relative perturbations across
    parameter groups with different bounds.
    """
    g = ADAPTER_GROUPS
    y = vec.copy()

    # Per-segment sigma scaling so genes with different bounds get proportionally-scaled noise
    segments = [
        (0 * g, 1 * g, ADAPTER_BOUNDS_PER_GROUP["conv_scale"]),
        (1 * g, 2 * g, ADAPTER_BOUNDS_PER_GROUP["conv_shift"]),
        (2 * g, 3 * g, ADAPTER_BOUNDS_PER_GROUP["head_scale"]),
        (3 * g, 4 * g, ADAPTER_BOUNDS_PER_GROUP["head_shift"]),
    ]
    for start, end, (lo, hi) in segments:
        sigma = sigma_scale * (hi - lo)
        for i in range(start, end):
            if rng.random() < p:
                y[i] += rng.normal(0.0, sigma)

    return _clamp_adapters(y)


def crossover_blend_adapters(
    a: np.ndarray,
    b: np.ndarray,
    rng,
    alpha: float = 0.2,
) -> np.ndarray:
    """BLX-alpha crossover applied uniformly across all adapter genes,
    then per-segment clamping."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    span = hi - lo
    child = rng.uniform(lo - alpha * span, hi + alpha * span)
    return _clamp_adapters(child)


# Hooks: apply adapters to conv/head
def _find_last_conv_and_head(model: "nn.Module"):
    last_conv = None
    linear_layers: List["nn.Linear"] = []

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
        if isinstance(m, nn.Linear):
            linear_layers.append(m)

    if len(linear_layers) >= 2:
        # Last hidden Linear before the final 2-output layer.
        head = linear_layers[-2]
    elif len(linear_layers) == 1:
        head = linear_layers[-1]
    else:
        head = None

    return last_conv, head


def _detect_adapter_dims(controller) -> Tuple[int, int]:
    C, H = 64, 32  # safe defaults
    if hasattr(controller, "model") and nn is not None:
        last_conv, head = _find_last_conv_and_head(controller.model)
        if last_conv is not None and hasattr(last_conv, "out_channels"):
            C = int(last_conv.out_channels)
        if head is not None and hasattr(head, "out_features"):
            H = int(head.out_features)
    return C, H


def apply_adapters(controller, expanded: Dict[str, np.ndarray]):
    """
    Dynamically injects scale and shift transformations into the model.
    
    Uses PyTorch forward hooks to modify the outputs of the last Conv2d 
    and the last hidden Linear layer during the forward pass.
    
    Returns:
        Callable: A cleanup function to safely detach hooks after evaluation.
    """
    if torch is None or not hasattr(controller, "model"):
        return lambda: None

    model = controller.model
    last_conv, head = _find_last_conv_and_head(model)
    hooks = []

    if last_conv is not None:
        cs = torch.tensor(expanded["conv_scales"], dtype=torch.float32).view(1, -1, 1, 1)
        ch = torch.tensor(expanded["conv_shifts"], dtype=torch.float32).view(1, -1, 1, 1)

        def _hook_conv(_m, _inp, out):
            return out * cs.to(out.device) + ch.to(out.device)
        
        hooks.append(last_conv.register_forward_hook(lambda m, i, o: _hook_conv(m, i, o)))
    if head is not None:
        hs = torch.tensor(expanded["head_scales"], dtype=torch.float32).view(1, -1)
        hh = torch.tensor(expanded["head_shifts"], dtype=torch.float32).view(1, -1)

        def _hook_head(_m, _inp, out):
            return out * hs.to(out.device) + hh.to(out.device)

        hooks.append(head.register_forward_hook(lambda m, i, o: _hook_head(m, i, o)))

    def _remove():
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    return _remove


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Stage 2a: grouped adapters only, behavioral params fixed. "
            "Adapter architecture matched to Stage 2b for a clean ablation."
        )
    )
    ap.add_argument("--pretrained_npy", default=None,
                    help="Path to flat CNN weights (.npy/.npz) from hybrid GA.")
    ap.add_argument("--pretrained_json", default=None,
                    help="Optional JSON with cnn_weights / controller_params.")
    ap.add_argument("--fixed_bp_json", required=True, help="JSON file with fixed behavioral parameters for Stage 2a.")
    ap.add_argument("--sim_host", default=None)
    ap.add_argument("--sim_port", type=int, default=None)
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

    # Env config
    if args.sim_host:
        ENV_CONF["host"] = args.sim_host
    if args.sim_port:
        ENV_CONF["port"] = int(args.sim_port)

    # Fixed BP 
    if not os.path.isfile(args.fixed_bp_json):
        raise FileNotFoundError(
            f"--fixed_bp_json is required and must point to an existing JSON file. "
            f"Got: {args.fixed_bp_json}"
        )

    with open(args.fixed_bp_json, "r") as f:
        FIXED_BP = json.load(f)

    FIXED_BP = clamp_bp_dict(FIXED_BP)
    print(f"[INFO] Loaded fixed BP from {args.fixed_bp_json}")

    env = try_make_env(PRIMARY_ENV_ID, fallback_ids=FALLBACK_ENV_IDS, conf=ENV_CONF)
    if env is None:
        raise RuntimeError("Cannot open Donkey env. Is the simulator running?")

    try:
        _run_stage2a(env, args, run_id_int, effective_seed, rng)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_stage2a(env, args, run_id_int, effective_seed, rng):
    # Pretrained controller (frozen CNN) 
    genome = _load_best_genome_from_files(args.pretrained_npy, args.pretrained_json)
    if genome is None:
        raise RuntimeError(
            "Stage requires pretrained CNN weights. "
            "Provide --pretrained_npy and/or --pretrained_json."
        )

    init_errors: List[str] = []
    controller = None
    for sig in [
        lambda: Controller(genome=genome, config=ControllerConfig(), load_weights=True),
        lambda: Controller(genome=genome, load_weights=True),
        lambda: controller_cfg_genome(genome),
    ]:
        try:
            controller = sig()
            break
        except Exception as e:
            init_errors.append(str(e))
            controller = None
    if controller is None:
        raise RuntimeError("Controller init failed: " + " | ".join(init_errors))

    if torch is not None and hasattr(controller, "model"):
        controller.model.eval()
        for p in controller.model.parameters():
            p.requires_grad_(False)
        print("[INFO] CNN frozen (no gradient updates)")

    # Adapter mapper 
    g = ADAPTER_GROUPS
    detC, detH = _detect_adapter_dims(controller)
    conv_C = int(os.environ.get("ADAPTER_LAST_CONV_C", detC))
    head_H = int(os.environ.get("ADAPTER_HEAD_DIM", detH))
    print(
        f"[Adapters] Using last_conv_C={conv_C}, head_dim={head_H} "
        f"(detected {detC}/{detH})"
    )
    mapper = _AdapterMapper(g, conv_channels=conv_C, head_dim=head_H)

    # Logger & banner 
    logger = make_logger(CSV_PATH, SPACE_NAME, base_seed=SEED, effective_seed=effective_seed)
    print("\n" + "=" * 60)
    print("Stage 2a: Adapters only (BP fixed)")
    print("=" * 60)
    print(f"Population:        {POP}")
    print(f"Eval budget:       {EVAL_BUDGET}")
    print(f"K-repeats:         {K_REPEATS}")
    print(f"Adapter groups:    {g}")
    print(f"Adapter dim:       {ADAPTER_DIM}  (4*G)")
    print(f"BP source:         {args.fixed_bp_json}")
    print(f"CSV:               {CSV_PATH}")
    print("=" * 60 + "\n")

    pop = max(1, int(POP))
    max_generations = max(1, math.ceil(EVAL_BUDGET / pop))
    population: List[np.ndarray] = [_rand_adapters(rng) for _ in range(pop)]
    fitnesses: List[Optional[float]] = [None] * pop

    trial = 0
    best_so_far = -1e12

    def eval_one(adapter_vec: np.ndarray, trial_idx: int) -> float:
        expanded = mapper.split(adapter_vec)
        cleanup = apply_adapters(controller, expanded)
        try:
            fits: List[float] = []
            for r in range(K_REPEATS):
                s = SEED + int(RUN_ID) * 100000 + (trial_idx + 1) * 100 + r
                set_all_seeds(s)
                if hasattr(controller, "reset_turn_stabilizer"):
                    controller.reset_turn_stabilizer()
                fits.append(
                    evaluate_with_frozen_cnn(
                        env, controller, FIXED_BP,
                        max_steps=MAX_STEPS, seed=s,
                    )
                )
            fit = float(np.mean(fits))
        finally:
            cleanup()

        payload = {
            "bp_fixed": FIXED_BP,
            "adapters_groups": {
                "groups": g,
                "conv_scale": adapter_vec[0 * g:1 * g].tolist(),
                "conv_shift": adapter_vec[1 * g:2 * g].tolist(),
                "head_scale": adapter_vec[2 * g:3 * g].tolist(),
                "head_shift": adapter_vec[3 * g:4 * g].tolist(),
            },
        }

        nonlocal best_so_far
        best_so_far = logger(trial_idx + 1, RUN_ID, payload, fit)

        if os.environ.get("VERBOSE", "0") == "1":
            print(
                f"[trial {trial_idx:03d}] run={RUN_ID} "
                f"fit={fit:.3f} best={best_so_far:.3f}"
            )
        return fit

    # Initial population evaluation 
    print("[INFO] Initial population evaluation")
    for i in range(pop):
        if trial >= EVAL_BUDGET:
            break
        fitnesses[i] = eval_one(population[i], trial_idx=trial)
        trial += 1
        print(f"[GEN 0] ind={i + 1}/{pop} fit={fitnesses[i]:.2f} best={best_so_far:.2f}")

    def tournament() -> np.ndarray:
        idxs = rng.choice(len(population), size=min(3, len(population)), replace=False)
        best_idx = max(idxs, key=lambda j: fitnesses[j])
        return population[best_idx]

    gen = 1
    while trial < EVAL_BUDGET and gen <= max_generations:
        print(f"\n[INFO] Generation {gen}/{max_generations}")
        offspring: List[Tuple[np.ndarray, float]] = []

        while len(offspring) < pop and trial < EVAL_BUDGET:
            p1, p2 = tournament(), tournament()
            child = crossover_blend_adapters(p1, p2, rng, alpha=0.2)
            child = mutate_adapters(child, rng, sigma_scale=0.10, p=0.25)
            fit = eval_one(child, trial_idx=trial)
            trial += 1
            offspring.append((child, fit))

        pool = list(zip(population, fitnesses)) + offspring
        pool = [(v, f) for (v, f) in pool if f is not None]
        pool.sort(key=lambda x: x[1], reverse=True)
        pool = pool[:pop]
        population = [v for (v, _) in pool]
        fitnesses = [f for (_, f) in pool]

        print(
            f"[GEN {gen}] best={fitnesses[0]:.2f} "
            f"avg={float(np.mean(fitnesses)):.2f}"
        )
        gen += 1

    print(
        f"\n[Stage 2a] trials={trial}/{EVAL_BUDGET}, "
        f"population={pop}, generations~={max_generations}"
    )
    print(f"[Stage 2a] CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
