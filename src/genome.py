"""Genome representation for the hybrid GA driving system.

This module stores the genetic material used by the GA:
- flat CNN weights in Controller parameter order,
- bounded behavior parameters used by lane/turn logic,
- optional controller policy parameters,
- mutation/crossover metadata.

Default CNN initialization uses Kaiming (conv) / Xavier (linear), which produces
layer-aware narrow weight distributions appropriate for CNN evolution. A
uniform_legacy mode (weights from uniform(-1, 1)) is kept for ablation
"""

from __future__ import annotations

import copy
import math
import random
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from .controller import Controller

_PARAM_INDEX_CACHE: Dict[tuple, dict] = {}


def _architecture_fingerprint(model: torch.nn.Module, input_shape: tuple) -> tuple:
    """Return a hashable fingerprint for the Controller parameter layout."""
    param_shapes = tuple(tuple(p.shape) for _, p in model.named_parameters())
    return (tuple(input_shape), param_shapes)


def _build_param_index_map(genome, input_shape=(1, 60, 80)):
    """Build or retrieve flat-weight index ranges grouped by layer type.

    Returned ranges are slices into Genome.cnn_weights and match the
    parameter order used by Controller._load_weights_from_genome.
    """
    ctrl = Controller(genome, input_shape=input_shape, load_weights=False)
    model = ctrl.model

    key = _architecture_fingerprint(model, input_shape)
    if key in _PARAM_INDEX_CACHE:
        return _PARAM_INDEX_CACHE[key]

    name_to_range = {}
    pos = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        name_to_range[name] = (pos, pos + n)
        pos += n

    conv_slices = []
    linear_slices_in_order = []

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for suffix in (".weight", ".bias"):
                key_name = module_name + suffix
                if key_name in name_to_range:
                    conv_slices.append(name_to_range[key_name])

        elif isinstance(module, torch.nn.Linear):
            for suffix in (".weight", ".bias"):
                key_name = module_name + suffix
                if key_name in name_to_range:
                    linear_slices_in_order.append(name_to_range[key_name])

    info = {
        "conv_slices": conv_slices,
        "linear_slices": linear_slices_in_order,
    }
    _PARAM_INDEX_CACHE[key] = info
    return info


def ensure_layer_slices(genome, input_shape=(1, 60, 80), mode="linear") -> None:
    """Attach flat-weight mutation slices to a genome.

    Args:
        genome: Genome-like object with cnn_weights.
        input_shape: Controller input shape.
        mode: "linear", "conv", or "all".
    """
    info = _build_param_index_map(genome, input_shape=input_shape)

    if mode == "linear":
        ranges = info["linear_slices"]
    elif mode == "conv":
        ranges = info["conv_slices"]
    elif mode == "all":
        ranges = info["conv_slices"] + info["linear_slices"]
    else:
        raise ValueError("mode must be one of: 'linear', 'conv', 'all'")

    genome.layer_slices = [slice(start, end) for start, end in ranges]


# ---------------------------------------------------------------------------
# Behavior-parameter schema
# ---------------------------------------------------------------------------

# Order must match Controller._load_behavior_params_from_genome:
#   0: kp_off
#   1: kp_head
#   2: kp_curv
#   3: curv_thr_slow
#   4: lane_conf_thr
#   5: canny_scale
#   6: hough_threshold      integer-like
#   7: diag_roi_row_start
BP_BOUNDS = [
    (0.00, 0.80),
    (0.00, 0.40),
    (0.00, 0.50),
    (0.30, 1.50),
    (0.20, 0.80),
    (0.70, 1.30),
    (10.0, 30.0),
    (0.45, 0.75),
]

INT_PARAMS = {6}


def clamp_bp_inplace(bp: list) -> None:
    """Clamp behavior parameters to their physical bounds in place."""
    for i in range(min(len(bp), len(BP_BOUNDS))):
        lo, hi = BP_BOUNDS[i]

        try:
            value = float(bp[i])
        except (TypeError, ValueError):
            value = lo

        if not math.isfinite(value):
            value = lo

        if value < lo:
            value = lo
        elif value > hi:
            value = hi

        if i in INT_PARAMS:
            value = float(round(value))

        bp[i] = float(value)


MIN_LAYER_STD: float = 1e-3

# Mutation defaults. The GA scheduler may override these per Genome instance.
DEFAULT_MUTATE_GAUSSIAN: bool = True
DEFAULT_SCALE_BY_STD: bool = False
DEFAULT_WEIGHT_CLIP_K: float = 5.0
DEFAULT_CNN_INIT_MODE: str = "kaiming_xavier"  # "kaiming_xavier" | "uniform_legacy"
DEFAULT_WEIGHT_ABS_CLIP: float = 0.5

# Match the forced-mutation fallback by using global weight std when no gene is hit
# Use "slice" only if you intentionally want stricter layer-local scaling.
DEFAULT_FORCED_MUTATION_STD_MODE: str = "global_legacy"  # "global_legacy" | "slice"


class Genome:
    """Candidate solution for the hybrid GA driving system.

    Attributes:
        cnn_weights: Flat CNN parameter vector in Controller model order.
        behavior_params: Lane/turn behavior parameters with natural bounds.
        controller_params: Optional auxiliary controller policy parameters.
        fitness: Fitness assigned by evaluator.
        layer_slices: Optional flat-weight slices for layer-aware mutation.
    """

    def __init__(
        self,
        cnn_weight_count: Optional[int] = None,
        behavior_param_count: Optional[int] = None,
        *,
        _skip_init: bool = False,
        cnn_init_mode: Optional[str] = None,
    ):
        """Create a genome.

        Args:
            cnn_weight_count: Backward-compatible argument. The value is no
                longer trusted blindly; the Controller architecture is the
                source of truth for the actual number of CNN parameters.
            behavior_param_count: Number of behavior parameters. Defaults to len(BP_BOUNDS).
            _skip_init: Used internally for copying/crossover without random
                initialization.
            cnn_init_mode: Optional override for CNN initialization mode.
        """
        self._requested_cnn_weight_count = cnn_weight_count

        self.cnn_weights: list[float] = []
        self.behavior_params: list[float] = []
        self.controller_params: dict[str, Any] = {}
        self.fitness: Optional[float] = None
        self.layer_slices: Optional[list] = None

        # Instance-level mutation/init settings.
        self.mutate_gaussian: bool = DEFAULT_MUTATE_GAUSSIAN
        self.scale_by_std: bool = DEFAULT_SCALE_BY_STD
        self.weight_clip_k: float = DEFAULT_WEIGHT_CLIP_K
        self.weight_abs_clip: float = DEFAULT_WEIGHT_ABS_CLIP
        self.cnn_init_mode: str = cnn_init_mode or DEFAULT_CNN_INIT_MODE
        self.forced_mutation_std_mode: str = DEFAULT_FORCED_MUTATION_STD_MODE

        if _skip_init:
            return

        ctrl = Controller(genome=None, input_shape=(1, 60, 80), load_weights=False)
        total_params = sum(int(p.numel()) for p in ctrl.model.parameters())

        if (
            cnn_weight_count is not None
            and int(cnn_weight_count) > 0
            and int(cnn_weight_count) != total_params
        ):
            raise ValueError(
                f"cnn_weight_count={cnn_weight_count} does not match Controller "
                f"parameter count={total_params}. Check Controller architecture."
            )

        self.cnn_weights = self._init_cnn_weights(ctrl, total_params, self.cnn_init_mode)

        n_behavior = behavior_param_count or len(BP_BOUNDS)
        self.behavior_params = [
            random.uniform(*BP_BOUNDS[i]) if i < len(BP_BOUNDS) else 0.0
            for i in range(n_behavior)
        ]
        clamp_bp_inplace(self.behavior_params)

    @staticmethod
    def _init_cnn_weights(
        ctrl: Controller,
        total_params: int,
        mode: str,
    ) -> list[float]:
        """Initialize CNN weights according to the selected strategy."""
        if mode == "uniform_legacy":
            return [random.uniform(-1.0, 1.0) for _ in range(total_params)]

        if mode == "kaiming_xavier":
            for module in ctrl.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

                elif isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

            weights: list[float] = []
            for p in ctrl.model.parameters():
                weights.extend(
                    p.detach().cpu().numpy().reshape(-1).astype(float).tolist()
                )
            return weights

        raise ValueError(
            f"Unknown cnn_init_mode={mode!r}. "
            "Expected 'uniform_legacy' or 'kaiming_xavier'."
        )

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------
    def mutate(self, mutation_rate: float, mutation_strength: float, rng=None):
        """
        Mutate CNN weights and behavior parameters in place.

        Hybrid version:
    - keeps the older/sophisticated mutation structure;
    - supports Gaussian or Uniform mutation;
    - supports std-scaled mutation;
    - uses layer_slices if present, otherwise treats CNN as one global vector;
    - forces at least one CNN and one behavior mutation;
    - clips CNN weights with both std-based bound and absolute cap.
        """
        if rng is None:
            gen = np.random.default_rng()
        elif isinstance(rng, np.random.Generator):
            gen = rng
        else:
            gen = np.random.default_rng(rng)

        mutation_rate = float(mutation_rate)
        mutation_strength = float(mutation_strength)

        use_gauss = bool(getattr(self, "mutate_gaussian", DEFAULT_MUTATE_GAUSSIAN))
        scale_by_std = bool(getattr(self, "scale_by_std", DEFAULT_SCALE_BY_STD))
        clip_k = float(getattr(self, "weight_clip_k", DEFAULT_WEIGHT_CLIP_K))
        abs_cap = float(getattr(self, "weight_abs_clip", DEFAULT_WEIGHT_ABS_CLIP))

        # ---------------- CNN WEIGHTS ----------------
        w = np.asarray(getattr(self, "cnn_weights", []) or [], dtype=np.float32)
        n = w.size

        if n:
            slices_meta = getattr(self, "layer_slices", None)

            if slices_meta:
                if isinstance(slices_meta[0], tuple):
                    slices = [s for _, s in slices_meta]
                else:
                    slices = list(slices_meta)
            else:
                slices = [slice(0, n)]

            any_hit = False

            for sl in slices:
                layer = w[sl]
                layer_size = layer.size

                if layer_size == 0:
                    continue

                hits = gen.random(layer_size) < mutation_rate
                any_hit = any_hit or bool(hits.any())

                # Local std reference for this slice/layer.
                ws = max(MIN_LAYER_STD, float(layer.std()))

                if hits.any():
                    n_hits = int(hits.sum())

                    if use_gauss:
                        sigma = (
                            mutation_strength * ws
                            if scale_by_std
                            else mutation_strength / np.sqrt(3.0)
                        )
                        layer[hits] += gen.normal(0.0, sigma, n_hits).astype(np.float32)

                    else:
                        # Uniform branch.
                        # If scale_by_std=True, make uniform noise Kaiming-friendly.
                        # If False, use legacy absolute uniform noise.
                        half_range = (
                            mutation_strength * ws
                            if scale_by_std
                            else mutation_strength
                        )
                        layer[hits] += gen.uniform(
                            -half_range,
                            half_range,
                            n_hits,
                        ).astype(np.float32)

                # Clip whole slice/layer.
                # std_bound adapts to the slice scale;
                # abs_cap prevents uniform_legacy-style explosion.
                std_bound = clip_k * ws

                if abs_cap > 0:
                    bound = max(MIN_LAYER_STD, min(std_bound, abs_cap))
                else:
                    bound = max(MIN_LAYER_STD, std_bound)

                np.clip(layer, -bound, bound, out=layer)
                w[sl] = layer

            # Forced CNN mutation if no CNN weight was selected.
            if not any_hit:
                idx = int(gen.integers(0, n))

                forced_mode = getattr(
                    self,
                    "forced_mutation_std_mode",
                    DEFAULT_FORCED_MUTATION_STD_MODE,
                )

                if forced_mode == "global_legacy":
                    ws = max(MIN_LAYER_STD, float(w.std()))

                elif forced_mode == "slice":
                    forced_slice = next(
                        (sl for sl in slices if sl.start <= idx < sl.stop),
                        slice(0, n),
                    )
                    ws = max(MIN_LAYER_STD, float(w[forced_slice].std()))

                else:
                    raise ValueError(
                        f"Unknown forced_mutation_std_mode={forced_mode!r}. "
                        "Expected 'global_legacy' or 'slice'."
                    )

                if use_gauss:
                    sigma = (
                        mutation_strength * ws
                        if scale_by_std
                        else mutation_strength / np.sqrt(3.0)
                    )
                    w[idx] += float(gen.normal(0.0, sigma))

                else:
                    half_range = (
                        mutation_strength * ws
                        if scale_by_std
                        else mutation_strength
                    )
                    w[idx] += float(gen.uniform(-half_range, half_range))

                std_bound = clip_k * ws

                if abs_cap > 0:
                    bound = max(MIN_LAYER_STD, min(std_bound, abs_cap))
                else:
                    bound = max(MIN_LAYER_STD, std_bound)

                w[idx] = float(np.clip(w[idx], -bound, bound))

            self.cnn_weights = w.astype(float).tolist()

        # ---------------- BEHAVIOR PARAMS ----------------
        if getattr(self, "behavior_params", None):
            bp = np.asarray(self.behavior_params, dtype=np.float32)
            m = bp.size

            if m:
                hitb = gen.random(m) < mutation_rate

                # Force at least one behavior param mutation.
                if not hitb.any():
                    hitb[int(gen.integers(0, m))] = True

                n_hitb = int(hitb.sum())

                if use_gauss:
                    # Behavior params are physical values, so do not scale by CNN std.
                    # Variance-match uniform(-a, a): std = a / sqrt(3).
                    sigma_b = mutation_strength / np.sqrt(3.0)
                    bp[hitb] += gen.normal(0.0, sigma_b, n_hitb).astype(np.float32)

                else:
                    bp[hitb] += gen.uniform(
                        -mutation_strength,
                        mutation_strength,
                        n_hitb,
                    ).astype(np.float32)

                bp = bp.astype(float).tolist()
                clamp_bp_inplace(bp)
                self.behavior_params = bp

    def crossover(
        self,
        other: "Genome",
        best_bias: float = 0.7,
        rng=None,
    ) -> "Genome":
        """Recombine two parent genomes into a child.

        If both parents have fitness, each gene is inherited from the fitter
        parent with probability best_bias. If fitness is missing, genes are
        selected uniformly from either parent.
        """
        if not 0.0 <= best_bias <= 1.0:
            raise ValueError("best_bias must be in [0, 1]")

        if len(self.cnn_weights) != len(other.cnn_weights):
            raise ValueError("Cannot crossover genomes with different CNN weight lengths")

        if rng is None:
            gen = np.random.default_rng()
        elif isinstance(rng, np.random.Generator):
            gen = rng
        else:
            gen = np.random.default_rng(rng)

        self_fitness = getattr(self, "fitness", None)
        other_fitness = getattr(other, "fitness", None)

        if self_fitness is not None and other_fitness is not None:
            if self_fitness > other_fitness:
                best, worst = self, other
            elif other_fitness > self_fitness:
                best, worst = other, self
            else:
                best, worst = (self, other) if gen.random() < 0.5 else (other, self)

            def pick(a, b):
                return a if gen.random() < best_bias else b

        else:
            best, worst = self, other

            def pick(a, b):
                return a if gen.random() < 0.5 else b

        child = Genome(
            behavior_param_count=len(self.behavior_params),
            _skip_init=True,
        )

        # CNN weights
        child.cnn_weights = [0.0] * len(self.cnn_weights)
        for i in range(len(self.cnn_weights)):
            child.cnn_weights[i] = pick(best.cnn_weights[i], worst.cnn_weights[i])

        # Behavior params
        best_bp = list(getattr(best, "behavior_params", []) or [])
        worst_bp = list(getattr(worst, "behavior_params", []) or [])
        length = max(len(best_bp), len(worst_bp))

        if len(best_bp) < length:
            best_bp += [0.0] * (length - len(best_bp))
        if len(worst_bp) < length:
            worst_bp += [0.0] * (length - len(worst_bp))

        child.behavior_params = [
            pick(best_bp[i], worst_bp[i])
            for i in range(length)
        ]
        clamp_bp_inplace(child.behavior_params)

        # Metadata and auxiliary params are inherited from the selected best parent.
        child.controller_params = copy.deepcopy(getattr(best, "controller_params", {}) or {} )
        child.fitness = None
        child.layer_slices = copy.deepcopy(getattr(best, "layer_slices", None))
        child.mutate_gaussian = getattr(best, "mutate_gaussian", DEFAULT_MUTATE_GAUSSIAN)
        child.scale_by_std = getattr(best, "scale_by_std", DEFAULT_SCALE_BY_STD)
        child.weight_clip_k = getattr(best, "weight_clip_k", DEFAULT_WEIGHT_CLIP_K)
        child.cnn_init_mode = getattr(best, "cnn_init_mode", DEFAULT_CNN_INIT_MODE)
        child.forced_mutation_std_mode = getattr(
            best,
            "forced_mutation_std_mode",
            DEFAULT_FORCED_MUTATION_STD_MODE,
        )

        return child

    # ------ Helpers ------

    def set_fitness(self, fitness: float) -> None:
        self.fitness = float(fitness)

    @staticmethod
    def from_existing(genome: "Genome") -> "Genome":
        """Copy genetic material and metadata while resetting fitness."""
        copied = Genome(
            behavior_param_count=len(getattr(genome, "behavior_params", []) or []),
            _skip_init=True,
        )

        copied.cnn_weights = list(getattr(genome, "cnn_weights", []) or [])
        copied.behavior_params = list(getattr(genome, "behavior_params", []) or [])
        copied.controller_params = copy.deepcopy(
            getattr(genome, "controller_params", {}) or {}
        )
        copied.fitness = None

        for attr in (
            # Lamarckian / imitation metadata
            "imitation_status",
            "last_imitation_loss",
            "imitation_samples",
            "imitation_candidates",
            "imitation_quality",
            # Architecture / mutation state
            "layer_slices",
            "mutate_gaussian",
            "scale_by_std",
            "weight_clip_k",
            "cnn_init_mode",
            "forced_mutation_std_mode",
            # Controller policy params
            "turn_bias_gain",
            "log_std",
        ):
            if hasattr(genome, attr):
                try:
                    setattr(copied, attr, copy.deepcopy(getattr(genome, attr)))
                except Exception:
                    logging.warning("Genome.from_existing: failed to copy attribute %r", attr)

        return copied

    def __repr__(self) -> str:
        return (
            f"Genome(fitness={self.fitness}, "
            f"cnn_params={len(self.cnn_weights)}, "
            f"behavior_params={len(self.behavior_params)}, "
            f"cnn_init_mode={self.cnn_init_mode!r})"
        )