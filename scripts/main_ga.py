"""
GA  experiment runner for DonkeyCar.

This script initializes a DonkeyCar Gym environment, runs a genetic
algorithm over controller genomes saves the best genome, and prints generation-level statistics.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import math
import numpy as np
import cv2

import gym # gym_donkeycar registers environments against the legacy gym API.

try:
    import gym_donkeycar  
except ImportError as exc:
    raise ImportError(
        "gym_donkeycar is required. Install it with: pip install gym-donkeycar"
    ) from exc

from src.evolve_ga import AdaptiveConfig, EliteProtection, evolve


def _to_float(value, default: float = 0.0) -> float:
    """Convert a value to a finite float, falling back to default."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    return result if math.isfinite(result) else float(default)


def safe_save_best(genome, out_dir="genomes"):
    """
    Save the best genome in both portable JSON and NPY formats.

    Stored fields:
        cnn_weights
        behavior_params
        controller_params
        fitness
    """
    os.makedirs(out_dir, exist_ok=True)

    fit_val = _to_float(getattr(genome, "fitness", None), default=0.0)

    # merge-in top-level params if controller_params missing fields
    cp = dict(getattr(genome, "controller_params", {}) or {})
    if "turn_bias_gain" not in cp and hasattr(genome, "turn_bias_gain"):
        cp["turn_bias_gain"] = float(getattr(genome, "turn_bias_gain"))
    if "log_std" not in cp and hasattr(genome, "log_std"):
        ls = getattr(genome, "log_std", None)
        if ls is not None and len(ls) >= 2:
            cp["log_std"] = [float(ls[0]), float(ls[1])]

    data = {
        "cnn_weights": getattr(genome, "cnn_weights", None),
        "behavior_params": getattr(genome, "behavior_params", None),
        "controller_params": cp,
        "fitness": fit_val,
    }
    json_path = os.path.join(out_dir, "best_genome.json")
    npy_path = os.path.join(out_dir, "best_genome.npy")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    np.save(npy_path, data, allow_pickle=True)

    print(f"Saved best genome JSON: {json_path}")
    print(f"Saved best genome NPY:  {npy_path}")

    return json_path, npy_path


def summarize_stats(generation_stats):
    if not generation_stats:
        print("Empty generation_stats.")
        return
    last = generation_stats[-1]
    best = max(generation_stats, key=lambda g: g.get("best_fitness", float("-inf")))
    div = last.get("diversity", {})
    last_best = _to_float(last.get('best_fitness', 0.0))
    last_mean = _to_float(last.get('mean_fitness', 0.0))
    last_std = _to_float(last.get('std_fitness', 0.0))
    best_overall = _to_float(best.get('best_fitness', 0.0))

    print("\nTraining summary")
    print(f"Last generation: {last.get('generation')} | stage={last.get('stage')} | "
          f"best={last_best:.2f} | mean={last_mean:.2f}±{last_std:.2f}")
    print(f"Best generation: #{best.get('generation')} with best={best_overall:.2f}")
    print(f"Diversity(last): gen={_to_float(div.get('genetic', 0.0)):.3e} "
          f"beh={_to_float(div.get('behavioral', 0.0)):.3f} fit={_to_float(div.get('fitness', 0.0)):.3f}")
    print(f"last generation time: {_to_float(last.get('time', 0.0)):.1f}s\n")


def try_make_env(primary_id: str, conf: dict, fallback_ids=None):
    """
    Create a DonkeyCar Gym environment.

    Tries the primary environment ID first, then any fallback IDs. If all attempts
    fail, prints diagnostics and raises a RuntimeError.
    """
    fallback_ids = fallback_ids or []
    errors = []
    try:
        return gym.make(primary_id, conf=conf)
    except Exception as e:
        errors.append((primary_id, str(e)))

    # fallbacks
    for env_id in fallback_ids:
        try:
            print(f"…Trying fallback env id: {env_id}")
            return gym.make(env_id, conf=conf)
        except Exception as e:
            errors.append((env_id, str(e)))

    print("None of the env ids worked.")
    print("Errors:")
    for env_id, msg in errors:
        print(f"  - {env_id}: {msg}")

    print("\nAvailable environment IDs in gym registry:")
    try:
        # gym <=0.26
        all_ids = sorted(list(gym.envs.registry.keys()))
    except Exception:
        # gym >=0.26
        all_ids = []
        try:
            for spec in gym.envs.registry.values():
                all_ids.append(spec.id)
            all_ids = sorted(set(all_ids))
        except Exception:
            all_ids = ["<Couldnt load ids>"]
    for eid in all_ids:
        if "donkey" in eid.lower():
            print("  •", eid)

    raise RuntimeError("Could not find/open Donkey environment. Check the environment ID and gym_donkeycar installation.")

def ensure_numeric_fitness(genome, generation_stats):
    """Ensure genome.fitness is a finite numeric value, recovering it from stats if needed."""

    fitness = _to_float(getattr(genome, "fitness", None), default=float("nan"))
    if math.isfinite(fitness):
        return fitness

    best_from_stats = max(
        (
            _to_float(g.get("best_fitness"), default=float("-inf"))
            for g in (generation_stats or [])
            if isinstance(g, dict)
        ),
        default=float("-inf"),
    )

    new_fit = best_from_stats if math.isfinite(best_from_stats) else 0.0

    if hasattr(genome, "set_fitness"):
        genome.set_fitness(new_fit)
    else:
        setattr(genome, "fitness", new_fit)

    print(f"Best genome fitness was missing or invalid; set to {new_fit:.2f}")
    return new_fit

def _run_training(env, cfg):
    """
    Run GA training on an already-created environment and save the best genome.

    Kept as a separate function (rather than inline in __main__) so the caller
    can wrap it in try/finally and guarantee env.close() runs even if evolve()
    or the post-processing steps raise.
    """
    result = evolve(
        env,
        config=cfg,
        max_generations=15,
        save_best_path="genomes/best_genome.json",
        )

    cv2.destroyAllWindows()
    if isinstance(result, tuple) and len(result) == 2:
        best_genome, generation_stats = result
    else:
        best_genome, generation_stats = result, []

    if best_genome is None:
        raise RuntimeError("evolve() returned None for best_genome. Check src.evolve_ga.evolve().")

    ensure_numeric_fitness(best_genome, generation_stats)
    safe_save_best(best_genome, out_dir="genomes")
    summarize_stats(generation_stats)


if __name__ == "__main__":
    # DonkeyCar simulator configuration.
    conf = {
        "host": os.environ.get("SIM_HOST", "127.0.0.1"),  # Override with SIM_HOST env var if needed.
        "port": int(os.environ.get("SIM_PORT", "9091")),
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": ".",
        "font_size": 100,
    }

    primary_id = "donkey-warehouse-v0"
    fallback_ids = ["donkey-warren-track-v0", "donkey-empty-v0"]

    cfg = AdaptiveConfig()
    cfg.elite_eval_runs = 2
    cfg.elite_accept_margin = 0.0
    cfg.elite_protection = EliteProtection(top_k=0)

    env = try_make_env(primary_id, conf, fallback_ids)

    try:
        _run_training(env, cfg)
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"Warning: failed to close environment cleanly: {exc}")