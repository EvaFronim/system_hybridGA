#!/usr/bin/env python3
"""Deterministic evaluation runner for saved GA genomes in DonkeySim.

Loads a saved genome, restores its CNN weights and behavior parameters,
runs deterministic evaluation episodes in a DonkeyCar simulator environment,
and exports summary metrics and per-step telemetry to CSV files.
"""

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Tuple

import gym
import numpy as np

try:
    import gym_donkeycar  # noqa: F401  # Registers DonkeyCar gym environments.
except ImportError:
    # DonkeyCar environments may already be registered in some setups.
    pass

from src.controller import Controller
from src.evolve_ga import (
    HyperParams,
    _auto_canny_from_sample,
    _choose_episode_aug,
    _ensure_bp,
    _sync_lane_hp_from_genome,
    _turn_info_from_edges,
    calculate_progressive_throttle_v2,
    detect_dashed_centerline,
    detect_diagonal_angle,
    detect_symmetric_midline,
    detect_turn_signals,
    lane_polyfit_features,
    load_best_genome,
    preprocess,
    reset_episode_calibration,
    seed_everything,
)


def make_env(name: str, host: str, port: int) -> Any:
    """Create a DonkeySim gym environment.
    Args:
        name: Gym environment ID, e.g. donkey-minimonaco-track-v0.
        host: DonkeySim host address.
        port: DonkeySim TCP port.
    Returns:
        A configured DonkeyCar gym environment.
    Raises:
        RuntimeError: If the environment cannot be created.
    """
    config = {"host": host, "port": int(port)}

    try:
        return gym.make(name, conf=config)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create DonkeySim environment '{name}' "
            f"at {host}:{port}."
        ) from exc

def build_controller(
    hp: HyperParams,
    genome_path: str,
    debug: bool = False,
) -> Tuple[Controller, dict]:
    """Load a saved genome and instantiate a Controller for evaluation."""

    # Apply training-time metadata to runtime hyperparameters so preprocessing,
    # controller input shape, and gains match the setup used during training.
    try:
        with open(genome_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        meta_hp = meta.get("hp", {}) if isinstance(meta, dict) else {}

        if isinstance(meta_hp, dict):
            for key, value in meta_hp.items():
                if not hasattr(hp, key):
                    continue

                try:
                    setattr(hp, key, value)
                except Exception as exc:
                    if debug:
                        print(f"(dbg) failed to set hp.{key}={value!r}: {exc}")

        input_shape = meta.get("input_shape") if isinstance(meta, dict) else None
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
            try:
                hp.resize_h = int(input_shape[1])
                hp.resize_w = int(input_shape[2])
            except (TypeError, ValueError) as exc:
                if debug:
                    print(f"(dbg) invalid input_shape={input_shape!r}: {exc}")

        if debug:
            print(f"(dbg) applied genome metadata; input HxW={hp.resize_h}x{hp.resize_w}")

    except (OSError, json.JSONDecodeError) as exc:
        if debug:
            print(f"(dbg) no readable genome metadata from {genome_path}: {exc}")

    # Use a dummy controller to compute the CNN parameter count expected by
    # the current controller architecture.
    dummy = Controller(
        input_shape=(1, hp.resize_h, hp.resize_w),
        load_weights=False,
    )
    expected_params = sum(param.numel() for param in dummy.model.parameters())
    genome = load_best_genome(expected_param_count=expected_params, path=genome_path)
    cnn_weights = getattr(genome, "cnn_weights", None) if genome is not None else None

    if genome is None:
        raise RuntimeError(f"Failed to load genome from {genome_path}")

    if cnn_weights is None:
        raise RuntimeError(f"Genome does not contain CNN weights: {genome_path}")

    if len(cnn_weights) != expected_params:
        raise RuntimeError(
            f"Genome CNN weight count mismatch for {genome_path}: "
            f"expected {expected_params}, got {len(cnn_weights)}"
        )

    # Legacy compatibility: older genomes may store controller parameters at
    # the top level instead of inside controller_params.
    controller_params = dict(getattr(genome, "controller_params", {}) or {})

    if "turn_bias_gain" not in controller_params and hasattr(genome, "turn_bias_gain"):
        controller_params["turn_bias_gain"] = float(getattr(genome, "turn_bias_gain"))

    if "log_std" not in controller_params and hasattr(genome, "log_std"):
        log_std = getattr(genome, "log_std", None)

        if isinstance(log_std, (list, tuple)) and len(log_std) >= 2:
            controller_params["log_std"] = [
                float(log_std[0]),
                float(log_std[1]),
            ]

    genome.controller_params = controller_params

    _ensure_bp(genome)
    _sync_lane_hp_from_genome(hp, genome)

    # Instantiate the real controller with the loaded genome weights.
    controller = Controller(
        genome=genome,
        input_shape=(1, hp.resize_h, hp.resize_w),
        load_weights=True,
    )

    if debug:
        controller.debug_weight_stats(top_k=20)

    if hasattr(controller, "reset_turn_stabilizer"):
        controller.reset_turn_stabilizer()

    try:
        info = controller.debug_state()
    except Exception as exc:
        if debug:
            print(f"(dbg) controller state unavailable: {exc}")
        info = {}

    return controller, info

def _get_first(info: dict, keys):
    for k in keys:
        if k in info and info[k] is not None:
            return info[k]
    return None


def deterministic_eval_loop(env, ctrl, hp, max_steps, max_throttle, stage_name, debug,
                            lane_cte_thr: float, road_cte_thr: float):
    """Run a single deterministic evaluation episode.

    Resets the environment, performs a brief auto-Canny calibration over the
    first few frames, then loops the full control stack (CNN → EMA → lane
    correction → throttle governor → Hough/dashed/symmetric aids) until the
    environment terminates or max_steps is reached.
    """
    # ─── Episode-level accumulators ─────────────────────────────────────────
    t0 = time.time()
    abs_cte_vals = []
    dsteer_vals = []
    raw_rows = []
    prev_steer = None

    # ─── Env reset (handle both gym old/new API tuple shapes) ──────────────
    try:
        reset_out = env.reset()
        reset_episode_calibration(hp)
        _choose_episode_aug(hp)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    except Exception:
        obs = env.reset()
        reset_episode_calibration(hp)

    # ─── Auto-Canny calibration over the first few frames ──────────────────
    # Disable augmentations during calibration so the Canny thresholds we
    # learn reflect the clean simulator distribution. Restored in `finally`.
    old_aug = getattr(hp, 'aug_prob', 0.0)
    try:
        try:
            hp.aug_prob = 0.0
        except Exception:
            pass

        calib_frames = 3
        lo_acc, hi_acc, n = 0, 0, 0
        tmp_obs = obs
        for _ in range(calib_frames):
            img = preprocess(tmp_obs, hp)
            roi = img
            lo, hi = _auto_canny_from_sample(
                (roi * 255).astype(np.uint8),
                scale=float(getattr(hp, 'canny_scale', 1.0)),
            )
            lo_acc += lo
            hi_acc += hi
            n += 1

            # advance one frame to get a fresh sample
            try:
                tmp = env.step((0.0, 0.10))
                tmp_obs = tmp[0] if isinstance(tmp, tuple) else tmp
            except Exception:
                pass

        if n > 0:
            hp.canny_low = int(lo_acc / n)
            hp.canny_high = int(hi_acc / n)

    finally:
        try:
            hp.aug_prob = old_aug
        except Exception:
            pass

    steps = 0
    done = False

    # warm-up to get moving
    WARM_STEPS = 3
    WARM_THR = 0.12
    STUCK_WINDOW = 50
    stuck_speed_thr = 0.1  

    last_steer = 0.0
    consecutive_turns = 0
    velocity_history: List[float] = []
    steering_history: List[float] = []
    progress_vals = []
    distance_vals = []
    model_out0_vals = []   # raw model output (out[0])
    act_out_vals    = []   # after activation (e.g. tanh(out0/T))
    final_steer_vals = []  # final steer returned by predict()

    prev_xz = None
    dist_total = 0.0
    dist_on_road = 0.0
    dist_in_lane = 0.0

    speed_hist = []
    is_stuck = False
    did_complete = False

    turn_int_ema = 0.0
    printed = False
    info = {}
   
    while not done and steps < int(max_steps):
        image = preprocess(obs, hp)

        # Run controller (single forward pass internally) and pull raw model
        # outputs from the controller's diagnostic state instead of doing a
        # second forward pass.
        steer_raw, thr_raw = ctrl.predict(image)
        info_pi = getattr(ctrl, "last_policy_info", {}) or {}
        out0 = float(info_pi.get("raw_steer_logit", 0.0))
        act0 = float(info_pi.get("cnn_steer", 0.0))

        if debug and steps % 50 == 0:
            print(f"[{steps}] raw_steer_logit={out0:+.3f} cnn_steer={act0:+.3f}")

        # store raw CNN stats
        model_out0_vals.append(out0)
        act_out_vals.append(act0)
        final_steer_vals.append(float(steer_raw))

        _sig = detect_turn_signals(image, hp)
        fast_hit = (detect_diagonal_angle(image, hp) is not None)

        turn_info, turn_int_ema = _turn_info_from_edges(
            image, hp, prev_t_ema=turn_int_ema, diag_hit=fast_hit
        )

        # 1) Steering EMA
        alpha_keep = float(getattr(hp, 'steer_ema_keep', 0.50))
        steering = alpha_keep * last_steer + (1.0 - alpha_keep) * float(steer_raw)

        # 2) Lane polyfit correction
        off, head, curv, conf = lane_polyfit_features(image, hp)

        # Measure current lane width from the symmetric-midline detector (if enabled).
        if bool(getattr(hp, "use_dynamic_lane_width", False)):
            try:
                _w_sconf, _w_soff, _w_shead, _w_sdbg = detect_symmetric_midline(image, hp)
                if isinstance(_w_sdbg, dict) and _w_sdbg.get("ok", False):
                    hp.est_lane_w_px = float(_w_sdbg["width_px"])
            except Exception:
                pass

        # 1. Reliability Check: Ensure the lane detection meets the minimum confidence threshold
        if conf > float(getattr(hp, "lane_conf_thr", 0.50)):
            ref_w_cfg = getattr(hp, 'ref_lane_w_px', None)
            REF_W = float(ref_w_cfg) if ref_w_cfg is not None else (0.60 * hp.resize_w)
            CUR_W = float(getattr(hp, 'est_lane_w_px', REF_W))
            scale_w = float(REF_W / max(8.0, CUR_W))

            kp_off  = float(hp.lane_kp_off)  * scale_w
            kp_head = float(hp.lane_kp_head) * (scale_w ** 0.5)
            kp_curv = float(hp.lane_kp_curv) * (scale_w ** 1.2)

            steering -= hp.offset_sign * (kp_off  * float(off))
            steering -= hp.offset_sign * (kp_head * float(head))
            steering += kp_curv * float(np.clip(curv, -hp.curv_thr_cap, hp.curv_thr_cap))

        steering = float(np.clip(steering, -1.0, 1.0))

        # 3) Throttle governor

        last_thr = velocity_history[-1] if velocity_history else float(max_throttle) * 0.5

        turn_info["lane_confidence"] = max(float(turn_info.get("lane_confidence", 0.0)),float(conf))
        turn_info["lane_curvature"] = (
            max(0.0, min(abs(float(curv)), 1.0))
            if conf > hp.lane_conf_thr
            else 0.0
        )


        final_thr = calculate_progressive_throttle_v2(
            raw_throttle=float(thr_raw),
            steering=float(steering),
            turn_info=turn_info,
            velocity_history=velocity_history,
            steering_history=steering_history,
            max_allowed=float(max_throttle),
            last_throttle=last_thr,
            consecutive_turns=consecutive_turns,
            stage_name=str(stage_name),
            hp=hp,
        )
        # Low-confidence fallback, aligned with training rollout
        try:
            lane_conf = float(turn_info.get("lane_confidence", 0.0))
        except Exception:
            lane_conf = 0.0

        if lane_conf < 0.15:
            final_thr = min(final_thr, float(getattr(hp, "safe_thr", 0.10)))
            steering *= 0.90
            steering = float(np.clip(steering, -1.0, 1.0))

        # 4) Warm-up throttle: force a minimum to overcome static friction at start
        if steps < WARM_STEPS:
            final_thr = max(final_thr, WARM_THR)

        final_thr = float(np.clip(final_thr, 0.0, 1.0))

        # 5) Hough helper applied AFTER the throttle governor (matches training pipeline)
        try:
            if bool(getattr(hp, 'use_hough_helper', True)):
                dir_sign = float(_sig.get('turn_direction', 0.0))
                conf_h = (float(_sig.get('direction_confidence', 0.0)) * float(_sig.get('overall_confidence', 0.0)))

                if conf_h > 0.45 and turn_int_ema > 0.30 and abs(dir_sign) > 0.0:
                    steering += (
                        float(getattr(hp, 'hough_kp_dir', 0.50))
                        * float(getattr(hp, 'steer_sign', 1))
                        * float(getattr(hp, 'dir_sign_gain', 1.0))
                        * dir_sign
                        * min(1.0, turn_int_ema)
                    )

                    # Enforce a minimum turn magnitude when a confident turn is detected
                    m = float(getattr(hp, 'min_turn_steer', 0.25))
                    if abs(steering) < m:
                        steering = (
                            1.0
                            if (float(getattr(hp, 'steer_sign', 1)) * dir_sign) > 0
                            else -1.0
                        ) * m

                steering = float(np.clip(steering, -1.0, 1.0))
        except Exception as e:
            if debug:
                print(f"(dbg) Hough helper failed at step {steps}: {e}")

        # 6) Dashed centerline / symmetric midline aids (applied after Hough)
        try:
            _dscore, _doff = detect_dashed_centerline(image, hp)
            _sconf, _soff, _shead, _ = detect_symmetric_midline(image, hp)

            if bool(getattr(hp, 'use_dash_control', True)) and float(_dscore) > float(getattr(hp, 'dash_score_thr', 0.45)):
                steering -= hp.offset_sign * hp.dash_kp * _doff

            if (
                bool(getattr(hp, 'use_sym_control', True))
                and float(_sconf) > float(getattr(hp, 'sym_conf_thr', 0.60))
                and abs(float(_soff)) < float(getattr(hp, 'sym_max_abs_off', 0.50))
                and abs(float(_shead)) < float(getattr(hp, 'sym_max_abs_head', 0.20))
            ):
                steering -= hp.offset_sign * hp.sym_kp_off * _soff
                steering -= hp.offset_sign * hp.sym_kp_head * _shead

            steering = float(np.clip(steering, -1.0, 1.0))

        except Exception as e:
            if debug:
                print(f"(dbg) Dashed/symmetric aids failed at step {steps}: {e}")

        if abs(steering) > 0.3:
            consecutive_turns += 1
        else:
            consecutive_turns = max(0, consecutive_turns - 1)

        last_steer = steering

        act = [float(np.clip(steering, -1.0, 1.0)),float(np.clip(final_thr, 0.0, 1.0))]

        step_out = env.step(act)

        # 7) Update sliding-window histories AFTER the action is sent
        if len(velocity_history) >= 10:
            velocity_history.pop(0)
            steering_history.pop(0)

        velocity_history.append(final_thr)
        steering_history.append(steering)

        if debug and steps % 15 == 0:
            edge_act = float((image > 0.2).mean())
            print(f"[{steps}] steer_raw≈{steer_raw:+.3f} steer={steering:+.3f} "
                  f"thr_raw≈{thr_raw:+.3f} thr={final_thr:.3f} edge_act={edge_act:.3f}")


        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            raise RuntimeError(f"Unexpected env.step output: len={len(step_out) if isinstance(step_out, tuple) else 'not tuple'}")

        pos = info.get("pos") if isinstance(info, dict) else None
        speed = info.get("speed") if isinstance(info, dict) else None
        lap_count = info.get("lap_count", 0) if isinstance(info, dict) else 0

        x = z = None
        if isinstance(pos, (tuple, list)) and len(pos) >= 3:
            x = float(pos[0])
            z = float(pos[2])

        dist_step = 0.0
        if x is not None and z is not None:
            if prev_xz is not None:
                dx = x - prev_xz[0]
                dz = z - prev_xz[1]
                dist_step = float((dx*dx + dz*dz) ** 0.5)
            prev_xz = (x, z)

        dist_total += dist_step

        # ---- on_road / in_lane ----
        on_road = ""
        in_lane = ""

        # ---- stuck ----
        if speed is not None:
            try:
                sp = float(speed)
                speed_hist.append(sp)
                if len(speed_hist) > STUCK_WINDOW:
                    speed_hist.pop(0)
                
                is_stuck = (len(speed_hist) >= STUCK_WINDOW and 
                           (sum(speed_hist)/len(speed_hist)) < stuck_speed_thr)
            except Exception:
                pass


        # ---- completion ----
        try:
            did_complete = bool(int(lap_count) >= 1)
        except Exception:
            pass


        # ---- pull cte from info ----
        cte = info.get("cte", None) if isinstance(info, dict) else None

        if cte is not None:
            abs_cte = abs(float(cte))
            abs_cte_vals.append(abs_cte)
            on_road = int(abs_cte <= float(road_cte_thr))
            in_lane = int(abs_cte <= float(lane_cte_thr))
            if on_road == 1:
                dist_on_road += dist_step
            if in_lane == 1:
                dist_in_lane += dist_step

        try:
            cur_steer = float(steering)  
            if prev_steer is not None:
                dsteer_vals.append(abs(cur_steer - prev_steer))
            prev_steer = cur_steer
        except Exception:
            pass

        # ---- raw row per step (BC-compatible columns) ----
        try:
            raw_rows.append({
                "t_wall": float(time.time() - t0),
                "step": int(steps + 1),
                "user_mode": "pilot",
                "steering": float(steering),
                "pilot_angle": float(steer_raw),
                "pos_cte": float(cte) if cte is not None else "",
                "abs_cte": abs(float(cte)) if cte is not None else "",

                "pos_x": x if x is not None else "",
                "pos_y": z if z is not None else "",
                "pos_speed": float(speed) if speed is not None else "",
                "dist_step": float(dist_step),
                "on_road": on_road if on_road is not None else "",
                "in_lane": in_lane if in_lane is not None else "",
            })

        except Exception:
            pass

        if isinstance(info, dict):
            if debug:
                print("info keys:", list(info.keys())[:50])
                debug = False  


            p = _get_first(info, ("progress", "pos/progress"))
            if p is not None:
                try:
                    progress_vals.append(float(p))
                except Exception:
                    pass

            d = _get_first(info, ("distance", "pos/distance"))
            if d is not None:
                try:
                    distance_vals.append(float(d))
                except Exception:
                    pass


        steps += 1

    metrics = {
    "dist_total": dist_total,
    "dist_on_road": dist_on_road,
    "dist_in_lane": dist_in_lane,
    "is_stuck": is_stuck,
    "did_complete": did_complete,
    }
    def print_sat(name, arr):
        arr = np.array(arr, dtype=np.float32)
        print(f"\n[{name}]")
        print(f"  total steps   : {len(arr)}")
        print(f"  mean          : {arr.mean():+.4f}")
        print(f"  std           : {arr.std():.4f}")
        print(f"  |val|>0.95    : {(np.abs(arr)>0.95).mean()*100:.1f}%")
        print(f"  val > +0.95   : {(arr>0.95).mean()*100:.1f}%")
        print(f"  val < -0.95   : {(arr<-0.95).mean()*100:.1f}%")
        print(f"  -0.1<val<+0.1 : {(np.abs(arr)<0.1).mean()*100:.1f}%")

    if model_out0_vals:
        print_sat("SAT raw model out[0]", model_out0_vals)

    if act_out_vals:
        print_sat("SAT after activation", act_out_vals)

    if final_steer_vals:
        print_sat("SAT final steer (predict output)", final_steer_vals)

    return steps, done, abs_cte_vals, dsteer_vals, raw_rows, metrics




#['donkey-generated-roads-v0', 'donkey-warehouse-v0', 'donkey-avc-sparkfun-v0', 'donkey-generated-track-v0',
#  'donkey-mountain-track-v0', 'donkey-roboracingleague-track-v0', 'donkey-waveshare-v0', 
# 'donkey-minimonaco-track-v0', 'donkey-warren-track-v0', 'donkey-thunderhill-track-v0', 'donkey-circuit-launch-track-v0']

def main():
    """CLI entry point: parse args, build controller, run episodes, write CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="donkey-generated-track-v0") 
    parser.add_argument("--host", default=os.environ.get("SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--genome", default="best_genome.json")
    parser.add_argument("--stage", default="Basic", help="Only affects max_throttle via CLI")
    parser.add_argument("--max-throttle", type=float, default=-1.0)  
    parser.add_argument("--seed", type=int, default=42)           
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-csv", type=str, default="eval_ga.csv")
    parser.add_argument("--env-id", type=str, default=None) 
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--control-mode",default="hybrid",choices=["hybrid", "polyfit_only", "cnn_only_steering"])
    parser.add_argument("--dynamic-lane-width", action="store_true",
                         help="Enable measured-width steering gain scaling.")

    args = parser.parse_args()
    seed_everything(int(args.seed))


    hp = HyperParams()

    # --- Sync HPs & input shape from saved genome meta (if any)) ---
    try:
        with open(args.genome, "r") as f:
            payload = json.load(f)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        ishape = meta.get("input_shape", None)
        if isinstance(ishape, (list, tuple)) and len(ishape) == 3:
            _, mh, mw = ishape
            try:
                hp.resize_h = int(mh)
                hp.resize_w = int(mw)
            except Exception:
                pass
        meta_hp = meta.get("hp", {})
        if isinstance(meta_hp, dict):
            for k, v in meta_hp.items():
                try:
                    setattr(hp, k, v)
                except Exception:
                    pass
    except Exception:
        pass
    try: hp.aug_prob = 0.0
    except Exception: pass

    hp.use_dynamic_lane_width = bool(args.dynamic_lane_width)

    _stage_thr = {
        "Basic": 0.08,             
        "Early-Intermediate": 0.10,
        "Late-Intermediate": 0.12,
        "Advanced": 0.14,
        "Expert": 0.20,
    }

    meta_thr = None
    try:
        meta_thr = float(meta_hp.get("max_throttle"))  
    except Exception:
        meta_thr = None

    if float(args.max_throttle) >= 0:
        chosen_thr = float(args.max_throttle)
    elif meta_thr is not None:
        chosen_thr = meta_thr
    else:
        chosen_thr = float(_stage_thr.get(str(args.stage), 0.20))

    if args.debug:
        print(f"[dbg] chosen max_throttle = {chosen_thr:.3f} "
              f"(cli={args.max_throttle}, meta={meta_thr}, stage='{args.stage}')")


    # Build controller and show state
    ctrl, cinfo = build_controller(hp, args.genome, debug=args.debug)
    if args.debug and cinfo:
        print("Controller state:", cinfo)

    # Env
    base_seed = int(args.seed)
    env = make_env(args.env, args.host, args.port)

    fieldnames = [
        "system","control_mode","run_id","model_path","max_loops","steps","fail_reason","budget_reached",
        "wall_time_sec",
        "did_complete","is_stuck",  # MOVED HERE - right after wall_time_sec
        "mean_abs_cte","p95_abs_cte","max_abs_cte",
        "pct_on_road","pct_in_lane",
        "mean_abs_dsteer","p95_abs_dsteer",
        "raw_csv","lane_cte_thr","road_cte_thr",
        "dist_total","dist_on_road","dist_in_lane",
    ]   


    file_exists = os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0
    f = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    # CTE thresholds in DonkeySim world units (meters in the Unity scene).
    # `info['cte']` is the signed distance from the car to the lane
    # centerline, computed by the simulator from ground-truth positions.
    #   lane_thr = 0.5  → "in-lane"  : |CTE| ≤ 0.5 m  (within 50cm of center)
    #   road_thr = 2.0  → "on-road"  : |CTE| ≤ 2.0 m  (still inside track bounds)
    lane_thr = 0.5
    road_thr = 2.0


    for ep in range(int(args.episodes)):
        seed = base_seed + ep
        seed_everything(seed)

        t0 = time.time()
        steps, done, abs_cte_vals, dsteer_vals, raw_rows, metrics = deterministic_eval_loop(
            env=env, ctrl=ctrl, hp=hp,
            max_steps=int(args.max_steps),
            max_throttle=float(chosen_thr),
            stage_name=str(args.stage),
            debug=bool(args.debug),
            lane_cte_thr=float(lane_thr),
            road_cte_thr=float(road_thr),
        )


        wall = time.time() - t0

        max_steps = int(args.max_steps)
        budget_reached = (int(steps) >= max_steps)

        # fail_reason 
        if budget_reached:
            fail_reason = "budget_reached"
        elif bool(done):
            fail_reason = "env_done"
        else:
            fail_reason = "timeout"

        # CTE stats
        if abs_cte_vals:
            mean_abs_cte = float(np.mean(abs_cte_vals))
            p95_abs_cte  = float(np.percentile(abs_cte_vals, 95))
            max_abs_cte  = float(np.max(abs_cte_vals))
            pct_in_lane  = float(np.mean(np.array(abs_cte_vals) <= lane_thr))
            pct_on_road  = float(np.mean(np.array(abs_cte_vals) <= road_thr))
        else:
            mean_abs_cte = None
            p95_abs_cte  = None
            max_abs_cte  = None
            pct_in_lane  = None
            pct_on_road  = None

        # Steering jitter stats (abs Δsteer)
        if dsteer_vals:
            mean_abs_dsteer = float(np.mean(dsteer_vals))
            p95_abs_dsteer  = float(np.percentile(dsteer_vals, 95))
        else:
            mean_abs_dsteer = 0.0
            p95_abs_dsteer  = 0.0

        os.makedirs("eval_ga_raw", exist_ok=True)
        rid = args.run_id if args.run_id else f"ga_{seed}"

        if int(args.episodes) > 1:
            rid = f"{rid}_ep{ep:02d}"

        raw_path = os.path.join("eval_ga_raw", f"{rid}.csv")

        with open(raw_path, "w", newline="") as rf:
            w = csv.DictWriter(
                rf,
                fieldnames=[
                    "t_wall","step","user_mode","steering","pilot_angle","pos_cte","abs_cte",
                    "pos_x","pos_y","pos_speed","dist_step","on_road","in_lane",
                ],
                extrasaction="ignore",
            )
            w.writeheader()
            for r in raw_rows:
                w.writerow(r)


        row = {
            "system": "GA",
            "control_mode": str(args.control_mode),
            "run_id": rid,
            "model_path": args.genome,
            "max_loops": int(args.max_steps),
            "steps": int(steps),
            "fail_reason": fail_reason,
            "budget_reached": bool(budget_reached),
            "wall_time_sec": float(wall),
            
            # RIGHT AFTER wall_time_sec
            "did_complete": bool(metrics.get("did_complete", False)),
            "is_stuck": bool(metrics.get("is_stuck", False)),

            "mean_abs_cte": mean_abs_cte,
            "p95_abs_cte": p95_abs_cte,
            "max_abs_cte": max_abs_cte,

            "pct_on_road": pct_on_road,
            "pct_in_lane": pct_in_lane,

            "mean_abs_dsteer": mean_abs_dsteer,
            "p95_abs_dsteer": p95_abs_dsteer,

            "raw_csv": os.path.abspath(raw_path),
            "lane_cte_thr": float(lane_thr),
            "road_cte_thr": float(road_thr),
            
            "dist_total": float(metrics.get("dist_total", 0.0)),
            "dist_on_road": float(metrics.get("dist_on_road", 0.0)),
            "dist_in_lane": float(metrics.get("dist_in_lane", 0.0)),
        }

        writer.writerow(row)
        f.flush()


        if args.debug:
            print(f"[dbg] ep={ep} seed={seed} steps={steps} done={done} reason={row['fail_reason']}")

    f.close()

    try:
        env.close()
    except Exception:
        pass

    print(f"[OK] Wrote {args.out_csv} ({args.episodes} episodes)")


if __name__ == "__main__":
    main()
