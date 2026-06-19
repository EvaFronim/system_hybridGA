"""
Core Utilities and Shared Logic for Autonomous Driving Ablation Studies.
"""

import os
import numpy as np
import cv2
import csv
import json
import random
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import types  

import gym  # gym_donkeycar registers environments against the legacy gym API.

try:
    import gym_donkeycar  # noqa: F401  (import for side effect: env registration)
except ImportError as exc:
    raise ImportError(
        "gym_donkeycar is required. Install it with: pip install gym-donkeycar"
    ) from exc

from src.controller import Controller, ControllerConfig

try:
    import torch
except Exception:
    torch = None

# Common run settings
K_REPEATS = int(os.environ.get("K_REPEATS", "5"))
SEED = int(os.environ.get("SEED", "123"))
RUN_ID = os.environ.get("RUN_ID", "0")

EVAL_BUDGET = int(os.environ.get("EVAL_BUDGET", "100"))
POP = int(os.environ.get("POP", "14"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))

# Common optimizer label
OPT_NAME = "GA"

# Common simulator configuration
PRIMARY_ENV_ID = "donkey-generated-track-v0"
FALLBACK_ENV_IDS: List[str] = []

# NOTE: the default host below is a local WSL2 network address specific to
# the original development machine. Anyone running this elsewhere should
# set the SIM_HOST environment variable to their own simulator host.
ENV_CONF = {
    "host": os.environ.get("SIM_HOST", "127.0.0.1"),
    "port": int(os.environ.get("SIM_PORT", "9091")),
    "body_style": "donkey",
    "body_rgb": (128, 128, 128),
    "car_name": ".",
    "font_size": 100,
}

# Shared behavioral/preprocessing parameter schema
PARAM_KEYS = [
    "lane_kp_off",
    "lane_kp_head",
    "lane_kp_curv",
    "curv_thr_slow",
    "lane_conf_thr",
    "canny_scale",
    "hough_threshold",
    "diag_roi_row_start",
]

# NOTE: these bounds duplicate genome.BP_BOUNDS, Controller's hardcoded
# clamps, and evolve_ga._sync_lane_hp_from_genome. All four copies must be
# changed together, or different parts of the pipeline will silently
# disagree on valid behavior-parameter ranges.
BP_BOUNDS: List[Tuple[float, float]] = [
    (0.00, 0.80),  # lane_kp_off
    (0.00, 0.40),  # lane_kp_head
    (0.00, 0.50),  # lane_kp_curv
    (0.30, 1.50),  # curv_thr_slow
    (0.20, 0.80),  # lane_conf_thr
    (0.70, 1.30),  # canny_scale
    (10.0, 30.0),  # hough_threshold
    (0.45, 0.75),  # diag_roi_row_start
]

class DummyGenome:
    """
    Minimal genome wrapper used to load pretrained CNN weights through the
    Controller's existing genome-based initialization path.

    The Stage 1b evaluator does not use behavior_params directly. They are kept
    only for compatibility with Controller implementations that expect them to
    exist on the genome object.
    """

    def __init__(
        self,
        flat_weights: np.ndarray,
        controller_params: Optional[dict] = None,
        behavior_params: Optional[dict] = None,
    ):
        self.cnn_weights = np.asarray(flat_weights, dtype=np.float32).ravel().tolist()
        self.controller_params = controller_params or {}
        self.behavior_params = behavior_params or None


def controller_cfg_genome(genome):
    """
    Build a Controller by injecting a pretrained genome through ControllerConfig.

    This is a compatibility fallback for Controller implementations that read
    the genome from the config object instead of accepting it directly as a
    constructor argument.
        Args:
        genome: Genome-like object containing pretrained CNN weights and any
            optional controller/behavior parameters expected by Controller.

    Returns:
        A Controller instance initialized with the provided genome.
    """
    cfg = ControllerConfig()

    try:
        setattr(cfg, "genome", genome)

        if hasattr(cfg, "load_weights"):
            setattr(cfg, "load_weights", True)

        return Controller(config=cfg)

    except TypeError:
        return Controller(cfg)
    
def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x
    

def _as_float32_array(value) -> Optional[np.ndarray]:
    """Convert a supported weight container to a flat float32 NumPy array."""
    if value is None:
        return None

    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        return None

    return arr.ravel()

def midpoint_bp_dict():
    mids = np.array(
        [(lo + hi) / 2.0 for lo, hi in BP_BOUNDS],
        dtype=float,
    )
    return bp_vec_to_dict(mids)

def _load_best_genome_from_files(
    npy_path: Optional[str],
    json_path: Optional[str],
) -> Optional[DummyGenome]:
    """
    Load pretrained CNN weights from NPY/NPZ and/or JSON files.

    Supported formats:
        1. A raw NumPy array containing flattened CNN weights.
        2. A 0-dimensional NumPy object array containing a dictionary.
        3. An NPZ file with a `cnn_weights` entry.
        4. A JSON file with a `cnn_weights` field.

    If both NPY/NPZ and JSON are provided, CNN weights from the NumPy file take
    precedence. Missing controller_params or behavior_params may be filled from
    the JSON file.

    Args:
        npy_path: Path to a .npy or .npz file.
        json_path: Optional path to a JSON metadata file.

    Returns:
        A DummyGenome containing pretrained CNN weights, or None if no valid
        weights could be loaded.
    """
    flat_weights = None
    controller_params = None
    behavior_params = None

    if npy_path:
        if not os.path.exists(npy_path):
            print(f"[WARN] NPY/NPZ file does not exist: {npy_path}")
        else:
            try:
                loaded = np.load(npy_path, allow_pickle=True)

                # Case 1: NPZ archive.
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    if "cnn_weights" in loaded:
                        flat_weights = _as_float32_array(loaded["cnn_weights"])

                    if "controller_params" in loaded:
                        controller_params = loaded["controller_params"].item()

                    if "behavior_params" in loaded:
                        behavior_params = loaded["behavior_params"].item()

                    loaded.close()

                # Case 2: NPY array or object array.
                elif isinstance(loaded, np.ndarray):
                    if loaded.ndim == 0:
                        item = loaded.item()

                        if isinstance(item, dict):
                            flat_weights = _as_float32_array(item.get("cnn_weights"))
                            controller_params = item.get("controller_params")
                            behavior_params = item.get("behavior_params")
                        else:
                            flat_weights = _as_float32_array(item)
                    else:
                        flat_weights = _as_float32_array(loaded)

            except Exception as e:
                print(f"[WARN] Failed to load NPY/NPZ file '{npy_path}': {e}")
                flat_weights = None

    if json_path:
        if not os.path.exists(json_path):
            print(f"[WARN] JSON file does not exist: {json_path}")
        else:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if flat_weights is None:
                    flat_weights = _as_float32_array(data.get("cnn_weights"))

                if controller_params is None:
                    controller_params = data.get("controller_params")

                if behavior_params is None:
                    behavior_params = data.get("behavior_params")

            except Exception as e:
                print(f"[WARN] Failed to load JSON file '{json_path}': {e}")

    if flat_weights is None:
        print("[WARN] Could not extract valid cnn_weights from the provided files.")
        return None

    print(
        f"[INFO] Loaded pretrained CNN weights: "
        f"shape={flat_weights.shape}, dtype={flat_weights.dtype}"
    )

    return DummyGenome(
        flat_weights=flat_weights,
        controller_params=controller_params,
        behavior_params=behavior_params,
    )


def preprocess_observation(
    obs: np.ndarray,
    canny_scale: float,
    diag_roi_row_start: float,
    hough_threshold: int,
    lane_conf_thr: float,
) -> Tuple[np.ndarray, float]:
    """
    Convert a simulator observation into a normalized edge image.

    This preprocessing pipeline is shared across ablation stages to keep the
    vision input consistent. Depending on the stage, these parameters may be
    evolved or fixed.

    Steps:
        1. Convert the observation to grayscale.
        2. Crop a lower image region of interest.
        3. Detect edges using scaled Canny thresholds.
        4. Optionally reinforce line-like structures using probabilistic Hough lines.
        5. Apply a low-confidence filtering heuristic when raw edge density is low.
        6. Resize and normalize the edge image for the controller.

    Args:
        obs: Raw simulator observation, expected as HxWx3 RGB or HxW grayscale.
        canny_scale: Multiplier for the base Canny thresholds.
        diag_roi_row_start: Normalized vertical start of the ROI in [0, 1].
        hough_threshold: Minimum Hough votes required to accept a line segment.
        lane_conf_thr: Threshold applied to raw edge density before filtering.

    Returns:
        edge_img: Float32 array with shape (1, 60, 80), normalized to [0, 1].
        edge_density: Fraction of active Canny edge pixels in the ROI before Hough overlay.
    """
    if obs is None:
        raise ValueError("Observation cannot be None.")

    # Convert RGB observations to grayscale. If the observation has more than
    # one channel but is not RGB, fall back to the first channel.
    if obs.ndim == 3 and obs.shape[-1] == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    elif obs.ndim == 2:
        gray = obs
    elif obs.ndim == 3:
        gray = obs[..., 0]
    else:
        raise ValueError(f"Unsupported observation shape: {obs.shape}")

    H, W = gray.shape[:2]
    roi_start = int(np.clip(diag_roi_row_start, 0.0, 0.95) * H)
    roi_start = min(max(0, roi_start), H - 1)
    roi = gray[roi_start:, :]

    # Canny edges
    low, high = int(50*canny_scale), int(150*canny_scale)
    edges = cv2.Canny(roi, low, high)

    # Confidence proxy: fraction of active edge pixels in the raw Canny output.
    conf = float(np.mean(edges > 0))

    # Reinforce line-like structures using probabilistic Hough transform.
    if hough_threshold > 0:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold,
                                minLineLength=10, maxLineGap=8)
        if lines is not None:
            overlay = np.zeros_like(edges)
            for x1,y1,x2,y2 in lines.reshape(-1,4):
                cv2.line(overlay,(x1,y1),(x2,y2),255,1)
            edges = cv2.bitwise_or(edges, overlay)

    # Low edge density triggers a simple filtering heuristic. Median filtering may
    # remove isolated noise, but it can also suppress thin lane edges.
    if conf < lane_conf_thr:
        edges = cv2.medianBlur(edges, 3)

    # Resize & normalize
    edges_resized = cv2.resize(edges, (80, 60), interpolation=cv2.INTER_AREA)
    edges_norm = (edges_resized.astype(np.float32) / 255.0)[np.newaxis, :]
    return edges_norm, conf


def reset_env_compat(env, seed: Optional[int] = None) -> tuple[Any, dict]:
    """
    Reset a Gym/Gymnasium environment in a version-agnostic way.

    Returns:
        (obs, info)
    """
    try:
        reset_out = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            try:
                env.seed(seed)
            except Exception:
                pass
        reset_out = env.reset()

    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
        if info is None:
            info = {}
        return obs, info

    return reset_out, {}


def step_env_compat(env, action):
    """
    Step a Gym/Gymnasium environment in a version-agnostic way.

    Normalizes the return signature to:
        (obs, reward, terminated, truncated, info)

    Older Gym:
        env.step(action) -> (obs, reward, done, info)

    Gymnasium / Gym >= 0.26:
        env.step(action) -> (obs, reward, terminated, truncated, info)
    """
    step_out = env.step(action)

    if not isinstance(step_out, tuple):
        raise TypeError(f"Expected env.step() to return a tuple, got {type(step_out).__name__}")

    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, bool(done), False, info

    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, reward, bool(terminated), bool(truncated), info

    raise ValueError(f"Unexpected env.step() return length: {len(step_out)}")

def try_make_env(primary_id: str, conf: dict, fallback_ids: Optional[List[str]] = None):
    """
    Create a DonkeyCar Gym environment.

    Tries the primary environment ID first, then any fallback IDs. If all
    attempts fail, prints diagnostics listing available Donkey environment IDs
    in the gym registry and raises a RuntimeError.

    Args:
        primary_id: The preferred Gym environment ID (e.g. "donkey-generated-track-v0").
        conf: Simulator configuration dict passed to gym.make().
        fallback_ids: Optional list of alternative env IDs to try if the primary fails.

    Returns:
        A constructed Gym environment instance.
    """
    fallback_ids = fallback_ids or []
    errors = []

    try:
        return gym.make(primary_id, conf=conf)
    except Exception as e:
        errors.append((primary_id, str(e)))

    for env_id in fallback_ids:
        try:
            print(f"…trying fallback env id: {env_id}")
            return gym.make(env_id, conf=conf)
        except Exception as e:
            errors.append((env_id, str(e)))

    print("None of the env ids worked.")
    print("Errors:")
    for env_id, msg in errors:
        print(f"  - {env_id}: {msg}")

    print("\nAvailable Donkey environment IDs in gym registry:")
    try:
        # gym <= 0.26
        all_ids = sorted(list(gym.envs.registry.keys()))
    except Exception:
        # gym >= 0.26
        all_ids = []
        try:
            for spec in gym.envs.registry.values():
                all_ids.append(spec.id)
            all_ids = sorted(set(all_ids))
        except Exception:
            all_ids = ["<could not load ids>"]

    for eid in all_ids:
        if "donkey" in eid.lower():
            print("  •", eid)

    raise RuntimeError(
        "Could not find/open Donkey environment. "
        "Check the environment ID and gym_donkeycar installation."
    )

def set_all_seeds(s: int):
    """Seed Python/NumPy/PyTorch RNGs.

    Note: this function does not set PYTHONHASHSEED and does not force
    deterministic cuDNN behavior (unlike seed_everything in evolve_ga.py).
    That only matters for GPU runs; on CPU (as used for all experiments in
    this project) it has no effect.
    """
    random.seed(s)
    np.random.seed(s)
    if torch is not None:
        try:
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass

def make_logger(csv_path, space_name, optimizer=OPT_NAME, base_seed=None, effective_seed=None):
    header = [
        "ts",
        "optimizer",
        "space",
        "trial",
        "run_id",
        "base_seed",
        "effective_seed",
        "params_json",
        "fitness",
        "best_so_far",
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    best = -1e12

    def log(trial_idx, run_id, params, fitness):
        nonlocal best
        best = max(best, fitness)

        row = [
            datetime.utcnow().isoformat() + "Z",
            optimizer,
            space_name,
            int(trial_idx),
            int(run_id),
            int(base_seed) if base_seed is not None else "",
            int(effective_seed) if effective_seed is not None else "",
            json.dumps(jsonable(params)),
            f"{float(fitness):.6f}",
            f"{best:.6f}",
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        return best

    return log

def rand_bp_vec(rng):
    return np.array([rng.uniform(lo, hi) for (lo, hi) in BP_BOUNDS], dtype=np.float32)

def clamp_bp_vec(x):
    out = np.empty_like(x)
    for i,(lo,hi) in enumerate(BP_BOUNDS):
        out[i] = min(hi, max(lo, float(x[i])))
    return out

def clamp_bp_dict(d: Dict[str, float]) -> Dict[str, float]:
    """
    Clamp each behavioral parameter in a dict to its BP_BOUNDS range.
    
    Preserves the integer-rounding contract for hough_threshold to stay
    consistent with bp_vec_to_dict.
    
    Args:
        d: Dict with PARAM_KEYS as keys and float values.
    
    Returns:
        New dict with values clamped to bounds. Missing keys are filled
        with the bound midpoint.
    """
    out = {}
    for i, key in enumerate(PARAM_KEYS):
        lo, hi = BP_BOUNDS[i]
        val = d.get(key, (lo + hi) / 2.0)
        out[key] = float(min(hi, max(lo, float(val))))
    out["hough_threshold"] = int(round(out["hough_threshold"]))
    return out

def bp_vec_to_dict(vec):
    d = {k: float(v) for k, v in zip(PARAM_KEYS, vec.tolist())}
    d["hough_threshold"] = int(round(d["hough_threshold"]))  # integer gene
    return d

def mutate_bp_vec(x, rng, sigma_scale=0.10, p=0.25):
    y = x.copy()
    for i,(lo,hi) in enumerate(BP_BOUNDS):
        if rng.random() < p:
            sigma = sigma_scale * (hi - lo)
            y[i] += rng.normal(0.0, sigma)
    return clamp_bp_vec(y)

def crossover_blend_bp_vec(a, b, rng, alpha=0.2):
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    span = hi - lo
    low  = lo - alpha * span
    high = hi + alpha * span
    child = rng.uniform(low, high)
    return clamp_bp_vec(child)



def evaluate_with_frozen_cnn(
    env,
    controller: Controller,
    params: Dict[str, float],
    max_steps: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Evaluate one parameter set using a frozen CNN controller with shared edge-image preprocessing.

    The controller receives the preprocessed edge image through Controller.predict(edge_img).
    Only preprocessing-related behavior parameters affect the image input directly.

    Active parameters:
        - lane_conf_thr
        - canny_scale
        - hough_threshold
        - diag_roi_row_start

    Inactive parameters, logged only for schema consistency:
        - lane_kp_off
        - lane_kp_head
        - lane_kp_curv
        - curv_thr_slow

    Args:
        env: DonkeyCar/Gym-compatible environment.
        controller: Frozen controller used to predict steering and throttle.
        params: Genome parameters as a dictionary.
        max_steps: Maximum number of simulator steps per rollout.
        seed: Optional environment seed for reproducible evaluation.

    Returns:
        Total accumulated simulator reward for the rollout.
    """
    if torch is None:
        raise ImportError("PyTorch is required to run the frozen CNN controller.")

    p = {k: float(params[k]) for k in PARAM_KEYS}

    # Gym >= 0.26 supports env.reset(seed=seed). Older Gym versions require
    # env.seed(seed) before reset().
    try:
        reset_out = env.reset(seed=seed)
    except TypeError:
        try:
            if seed is not None and hasattr(env, "seed"):
                env.seed(seed)
        except Exception:
            pass
        reset_out = env.reset()

    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {}

    total_reward = 0.0

    for _ in range(max_steps):
        # Only these four preprocessing genes affect Stage 1 behavior.
        edge_img, _ = preprocess_observation(
            obs,
            canny_scale=p["canny_scale"],
            diag_roi_row_start=p["diag_roi_row_start"],
            hough_threshold=int(round(p["hough_threshold"])),
            lane_conf_thr=p["lane_conf_thr"],
        )

        with torch.no_grad():
            steering, throttle = controller.predict(edge_img)

        # Keep throttle in a conservative range for stable simulator rollouts.
        throttle = float(np.clip(throttle, 0.05, 0.10))
        action = [float(steering), throttle]

        step_out = env.step(action)

        if len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated = bool(done)
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_out

        total_reward += float(reward)

        if terminated or truncated:
            break

    return float(total_reward)






class HybridIndividual:
    """
    Hybrid GA individual.

    Attributes:
        cnn_vec: Flat float32 vector of all CNN weights.
        bp_vec:  float32 vector of the 8 behavioral parameters.
        fitness: Mean (or trimmed-mean) reward from evaluate_with_frozen_cnn,
                 or None if the individual has not been evaluated yet.
    """
    def __init__(self, cnn_vec: np.ndarray, bp_vec: np.ndarray):
        self.cnn_vec = np.asarray(cnn_vec, dtype=np.float32)
        self.bp_vec = np.asarray(bp_vec, dtype=np.float32)
        self.fitness: Optional[float] = None

    def copy(self) -> "HybridIndividual":
        c = HybridIndividual(self.cnn_vec.copy(), self.bp_vec.copy())
        c.fitness = self.fitness
        return c


def mutate_cnn_vec(cnn: np.ndarray,
                   rng,
                   sigma: float = 0.01,
                   p: float = 0.05) -> np.ndarray:
    """
    Sparse Gaussian mutation on the CNN weight vector.

    Args:
        cnn:   Flat CNN weight vector.
        rng:   numpy Generator.
        sigma: Std of the additive Gaussian noise applied to selected weights.
        p:     Per-weight probability of being mutated.

    Returns:
        A new mutated vector. The input is not modified in place.
    """
    y = cnn.copy()
    mask = rng.random(y.shape) < p
    n = mask.sum()
    if n > 0:
        y[mask] += rng.normal(0.0, sigma, size=n)
    return y


def crossover_uniform_cnn(a: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    """
    Uniform crossover on CNN weights: for each weight, take parent A or B
    with equal probability.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    assert a.shape == b.shape
    mask = rng.random(a.shape) < 0.5
    return np.where(mask, a, b)


def build_controller_from_individual(
    ind: HybridIndividual,
    base_controller_params: Dict,
) -> Controller:
    """
    Build a frozen-CNN Controller from a HybridIndividual.

    Behavior parameters are NOT injected through the controller genome.
    They are passed explicitly to evaluate_with_frozen_cnn(...) by the caller.
    """
    genome_like = types.SimpleNamespace(
        cnn_weights=ind.cnn_vec.tolist(),
        behavior_params=None,
        controller_params=dict(base_controller_params or {}),
    )

    cfg = ControllerConfig()
    controller = Controller(genome=genome_like, config=cfg, load_weights=True)

    if hasattr(controller, "model"):
        controller.model.eval()

    return controller

def trimmed_mean(vals: List[float], trim_ratio: float = 0.10) -> float:
    """
    Compute a robust mean by discarding a fraction of the lowest and highest
    finite values before averaging.

    Used for:
        - per-individual fitness aggregation over repeated rollouts
        - robust per-generation fitness summaries

    For small sample sizes, if trim_ratio would remove zero values but at least
    five finite values are available, one value is removed from each side.
    """
    arr = [float(v) for v in vals if np.isfinite(float(v))]
    n = len(arr)

    if n == 0:
        return 0.0

    arr.sort()

    if n < 3:
        return float(np.mean(arr))

    k = int(np.floor(n * trim_ratio))

    if trim_ratio > 0.0 and k == 0 and n >= 5:
        k = 1

    k = min(k, n // 3)

    core = arr[k:n - k] if n - 2 * k > 0 else arr
    return float(np.mean(core))

