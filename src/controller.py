"""Neural controller for autonomous navigation.

This module defines a lightweight CNN-based controller with:
- deterministic inference,
- stochastic policy sampling for reinforcement learning,
- turn-direction stabilization,
- optional loading of CNN and auxiliary parameters from a genome.

The controller is designed for edge-map inputs and outputs steering and
throttle commands.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)

# Note: bp_kp_*, bp_curv_thr_slow, bp_lane_conf_thr, bp_canny_scale are
# stored for downstream consumers (lane fitter, governor) used in other
# stages of the project. They are not consumed by Controller methods directly.

@dataclass
class ControllerConfig:
    """Hyperparameters for the Controller and its components.

    Groups settings for the CNN backbone (channels, kernels, strides,
    hidden dim), the turn stabilizer (EMA, hysteresis, bypass), the
    stochastic policy (initial log_std and bounds), and the turn-bias
    gain. Defaults are tuned for 1x60x80 edge-map inputs.
    """

    # CNN architecture
    conv_channels: Tuple[int, ...] = (8, 16, 32)
    conv_kernels: Tuple[int, ...] = (5, 5, 3)
    conv_strides: Tuple[int, ...] = (2, 2, 2)
    hidden_dim: int = 50

    # Turn stabilizer (EMA + hysteresis + bypass)
    turn_alpha: float = 0.55
    switch_margin: float = 0.12
    bypass_threshold: float = 0.70
    bypass_hold_frames: int = 2

    # Stochastic policy
    initial_log_std: float = -1.2
    std_bounds: Tuple[float, float] = (1e-3, 2.0)

    # Turn bias gain
    initial_turn_bias_gain: float = 0.20
    turn_bias_bounds: Tuple[float, float] = (0.0, 0.5)

    # Compute device
    device: str = "cpu"

class AdaptiveCNN(nn.Module):
    """Small CNN backbone that auto-computes flattened size from input shape.
    Output is a 2-vector [steer_raw, throttle_raw] (pre-activation).
    """

    def __init__(self, input_shape: Tuple[int, int, int], config: ControllerConfig):
        super().__init__()
        self.input_shape = input_shape
        self.config = config

        if len(input_shape) != 3:
            raise ValueError(f"Expected input_shape (C, H, W), got {input_shape}")

        c, h, w = input_shape
        self.conv_layers = nn.ModuleList()
        in_ch = c
        for out_ch, k, s in zip(config.conv_channels, config.conv_kernels, config.conv_strides):
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s))
            in_ch = out_ch

        # Compute flattened size after all conv layers
        self.conv_output_size = self._calc_out_size(h, w)
        self.fc1 = nn.Linear(self.conv_output_size, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 2)  # [steer_raw, throttle_raw]

    def _calc_out_size(self, h: int, w: int) -> int:
        """Compute the flattened conv-output size for the given input H, W."""
        for k, s in zip(self.config.conv_kernels, self.config.conv_strides):
            h = (h - k) // s + 1
            w = (w - k) // s + 1
            if h <= 0 or w <= 0:
                raise ValueError(
                    f"Invalid conv output size: got H={h}, W={w}. "
                    f"Check input_shape={self.input_shape}, kernels={self.config.conv_kernels}, "
                    f"strides={self.config.conv_strides}."
                )
        return self.config.conv_channels[-1] * h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class TurnStabilizer:
    """Stabilize turn direction over frames using EMA + hysteresis + bypass.

    Maintains a previous (direction, confidence) pair and updates it from
    frame-by-frame detections. Three behaviors:

    Bypass: when the new detection is confident enough, accept it immediately and hold for
    bypass_hold_frames further frames.
    Hysteresis: when switching direction, require the new confidence
    to exceed the old one by at least switch_margin to avoid jitter.
    EMA decay: when no detection is available, slowly decay the
      previous confidence rather than dropping it abruptly.
    """

    def __init__(self, config: ControllerConfig):
        self.alpha = config.turn_alpha
        self.switch_margin = config.switch_margin
        self.bypass_threshold = config.bypass_threshold
        self.bypass_hold_frames = config.bypass_hold_frames
        self.prev_dir: Optional[str] = None
        self.prev_conf: float = 0.0
        self._bypass_left: int = 0

    def update(self, direction: Optional[str], confidence: float) -> Tuple[Optional[str], float]:
        """Update stabilized (direction, confidence) given a new detection."""
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # High-confidence detection: bypass smoothing
        if direction is not None and confidence >= self.bypass_threshold:
            self.prev_dir, self.prev_conf = direction, confidence
            self._bypass_left = self.bypass_hold_frames
            return self.prev_dir, self.prev_conf

        # Continue bypass for a few more frames, decaying confidence
        if self._bypass_left > 0:
            self._bypass_left -= 1
            self.prev_conf = max(0.0, self.prev_conf * 0.95)
            return self.prev_dir, self.prev_conf

        # First valid detection (no previous direction yet)
        if self.prev_dir is None and direction is not None:
            self.prev_dir, self.prev_conf = direction, confidence
            return self.prev_dir, self.prev_conf

        # No detection this frame: decay confidence, keep direction
        if direction is None:
            self.prev_conf = max(0.0, self.prev_conf * 0.9)
            return self.prev_dir, self.prev_conf

        # EMA on confidence
        ema_conf = self.alpha * confidence + (1 - self.alpha) * self.prev_conf

        # Hysteresis: reject direction switches unless confidence margin is met
        if self.prev_dir is not None and direction != self.prev_dir:
            if confidence < (self.prev_conf + self.switch_margin):
                self.prev_conf = ema_conf
                return self.prev_dir, self.prev_conf

        # Accept new direction
        self.prev_dir, self.prev_conf = direction, ema_conf
        return self.prev_dir, self.prev_conf

    def reset(self):
        """Reset stabilizer state (called between episodes)."""
        self.prev_dir, self.prev_conf, self._bypass_left = None, 0.0, 0

class Controller:
    """Main controller: CNN backbone + stochastic policy + turn stabilizer.

    The CNN produces 2 raw outputs (steer_raw, throttle_raw); a turn
    stabilizer (see :class: TurnStabilizer) applies a directional
    correction. The same backbone is
    used both for deterministic inference (predict) and for training (act).

    Auxiliary parameters loaded from genome.controller_params (if
    present): turn_bias_gain, log_std (2 values).
    """

    def __init__(
        self,
        genome: Optional[Any] = None,
        input_shape: Tuple[int, int, int] = (1, 60, 80),
        config: Optional[ControllerConfig] = None,
        load_weights: bool = True,
        use_internal_stabilizer: bool = True,
    ):
        self.config = config or ControllerConfig()
        self.input_shape = input_shape
        self.genome = genome
        self.use_internal_stabilizer = use_internal_stabilizer
        self.last_policy_info: Dict[str, Any] = {}
        self.device = torch.device(self.config.device)
        self.model = AdaptiveCNN(input_shape, self.config).to(self.device)


        # Behavior modulation dict (defaults are identity / no-op).
        # Kept as a public attribute so external components (governor,
        # evaluator, GA wrapper) can override values per-episode.
        self.behavior: Dict[str, float] = {
            "throttle_gain": 1.0,
            "steer_gain": 1.0,
            "center_bias": 0.0,
            "smoothness": 0.0,
            "turn_bias_boost": 0.0,
        }


        # Stochastic policy parameters (not part of the CNN's state_dict)
        self.log_std = nn.Parameter(
            torch.full((2,), self.config.initial_log_std, device=self.device))
        
        self.turn_bias_gain = nn.Parameter(
            torch.tensor(self.config.initial_turn_bias_gain, device=self.device))

        self.turn_stab = TurnStabilizer(self.config) if use_internal_stabilizer else None

        # Load weights / aux / behavior params from genome if provided
        if genome and load_weights:
            self._load_weights_from_genome()
            self._load_aux_params_from_genome()
            self._load_behavior_params_from_genome()

    def _load_weights_from_genome(self):
        """Copy genome.cnn_weights into the model parameters."""

        if not hasattr(self.genome, "cnn_weights"):
            raise ValueError("Genome lacks 'cnn_weights'")
        weights = self.genome.cnn_weights
        total = sum(p.numel() for p in self.model.parameters())
        if len(weights) != total:
            raise ValueError(f"Genome weights size {len(weights)} != model params {total}")
        with torch.no_grad():
            idx = 0
            for p in self.model.parameters():
                n = p.numel()
                p.copy_(torch.tensor(weights[idx:idx + n], device=self.device).view_as(p))
                idx += n

    def _load_aux_params_from_genome(self):
        """Load turn_bias_gain / log_std from genome.controller_params"""
        cp = getattr(self.genome, "controller_params", {}) or {}
        with torch.no_grad():
            # turn_bias_gain
            if "turn_bias_gain" in cp:
                val = float(cp["turn_bias_gain"])
                lo, hi = self.config.turn_bias_bounds
                val = float(np.clip(val, lo, hi))
                self.turn_bias_gain.copy_(torch.tensor(val, device=self.device))
            # log_std (must be exactly 2 values)
            if "log_std" in cp:
                ls = cp["log_std"]
                if hasattr(ls, "__len__") and len(ls) >= 2:
                    arr = np.array(ls[:2], dtype=np.float32)
                    lo, hi = np.log(self.config.std_bounds[0]), np.log(self.config.std_bounds[1])
                    arr = np.clip(arr, lo, hi)
                    self.log_std.copy_(torch.tensor(arr, device=self.device))
                else:
                    logger.warning(
                        "Skipping log_std load: expected ≥2 values, got %r", ls
                    )

    # -------- Training utilities --------

    def _load_behavior_params_from_genome(self):
        """Load behavior parameters from the genome (raw, not normalized).

        Expects genome.behavior_params to be a sequence of at least 8 values
        in this order:
            [0] kp_off, [1] kp_head, [2] kp_curv, [3] curv_thr_slow,
            [4] lane_conf_thr, [5] canny_scale, [6] hough_threshold,
            [7] diag_roi_row_start.

        Values are clamped to physical bounds. The Controller itself only uses
        indices 5-7 (in _detect_turn_direction); the rest are stored for
        downstream consumers (lane fitter, governor).
        """
        bp = getattr(self.genome, "behavior_params", None)
        if not bp or len(bp) < 8:
            self.bp_kp_off = 0.0
            self.bp_kp_head = 0.0
            self.bp_kp_curv = 0.0
            self.bp_curv_thr_slow = 1.0
            self.bp_lane_conf_thr = 0.55
            self.bp_canny_scale = 1.0
            self.bp_hough_threshold = 20
            self.bp_diag_roi_row_start = 0.50
            return

        # NOTE: these bounds duplicate genome.BP_BOUNDS. If you change one,
        # you must change the other, or Controller and Genome will silently
        # disagree on valid behavior-parameter ranges.
        def _clamp(x, lo, hi):
            try:
                x = float(x)
            except Exception:
                return float(lo)
            if x < lo: x = lo
            if x > hi: x = hi
            return float(x)

        self.bp_kp_off = _clamp(bp[0], 0.00, 0.80)
        self.bp_kp_head = _clamp(bp[1], 0.00, 0.40)
        self.bp_kp_curv = _clamp(bp[2], 0.00, 0.50)
        self.bp_curv_thr_slow = _clamp(bp[3], 0.30, 1.50)
        self.bp_lane_conf_thr = _clamp(bp[4], 0.20, 0.80)

        self.bp_canny_scale = _clamp(bp[5], 0.70, 1.30)
        self.bp_hough_threshold = int(round(_clamp(bp[6], 10.0, 30.0)))
        self.bp_diag_roi_row_start = _clamp(bp[7], 0.45, 0.75)
        
    def trainable_params(self):
        # Optimizer trains CNN params + log_std + turn_bias_gain
        return list(self.model.parameters()) + [self.log_std, self.turn_bias_gain]

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.log_std.data = self.log_std.data.to(self.device)
        self.turn_bias_gain.data = self.turn_bias_gain.data.to(self.device)
        self.config.device = str(self.device)
        return self

    def predict(self, image: np.ndarray) -> Tuple[float, float]:
        """Deterministic action (mean) with confidence-weighted CNN + stabilizer fusion.

        Returns the mean steer/throttle without sampling. The turn stabilizer
        adjusts the CNN steering output via a confidence-weighted bias, and
        the behavior dict applies post-hoc gain/bias modulation.
        """
        with torch.no_grad():
            x = self._prepare_input(image)
            out = self.model(x).squeeze(0)

            raw_steer = out[0]
            raw_throttle = out[1]

            cnn_steer = torch.tanh(raw_steer)
            cnn_throttle = torch.sigmoid(raw_throttle)

            # Default stabilizer values
            st_dir = None
            st_conf = 0.0
            stabilizer_delta = torch.tensor(0.0, device=self.device)

            # Teacher / stabilizer steering
            teacher_steer = cnn_steer.clone()

            if self.use_internal_stabilizer and self.turn_stab:
                dir_hint, conf = self._detect_turn_direction(image)
                st_dir, st_conf = self.turn_stab.update(dir_hint, conf)
                st_conf = float(np.clip(st_conf, 0.0, 1.0))

                if st_dir is not None and st_conf > 0.0:
                    base = self.turn_bias_gain.clamp(*self.config.turn_bias_bounds)
                    extra = torch.tensor(
                        self.behavior["turn_bias_boost"], device=self.device
                    )
                    direction_sign = 1.0 if st_dir == "right" else -1.0
                    stabilizer_delta = (base + extra) * st_conf * direction_sign

                    teacher_steer = (
                        cnn_steer + stabilizer_delta
                    ).clamp(-1.0, 1.0)

            # Behavioral scaling/bias on means
            steer_mean = (
                teacher_steer * self.behavior["steer_gain"]
                + self.behavior["center_bias"]
            ).clamp(-1.0, 1.0)
            throttle_mean = (
                cnn_throttle * self.behavior["throttle_gain"]
            ).clamp(0.0, 1.0)

            steering = float(steer_mean.item())
            throttle = float(throttle_mean.item())
        self.last_policy_info = {
            "teacher_steer":     float(teacher_steer.item()),
            "stabilizer_delta":  float(stabilizer_delta.clamp(-1.0, 1.0).item()),
            "st_conf":           float(st_conf),
            "st_dir":            st_dir,
            "raw_steer_logit":   float(raw_steer.item()),
            "cnn_steer":         float(cnn_steer.item()),
        }
        return steering, throttle

    def act(
        self,
        image: np.ndarray,
        explore: bool = True,
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """Stochastic action for REINFORCE-style training.

        If explore=True, samples steer/throttle from a Normal distribution
        around the policy means and returns the log-probability and entropy
        needed for policy-gradient updates. If explore=False, returns the
        deterministic means with zero log-prob and entropy.

        Note: The returned log_prob and entropy are computed on the pre-clamp sample. 
        The action returned to the environment is clamped to [-1, 1] (steer) and [0, 1] (throttle), 
        so the gradient signal corresponds to the unclamped distribution.
        """
        x = self._prepare_input(image)
        self.model.train()
    
        out = self.model(x).squeeze(0)
        steer_mean = torch.tanh(out[0])
        throttle_mean = torch.sigmoid(out[1])

        # Stabilizer turn-bias correction (before sampling)
        if self.use_internal_stabilizer and self.turn_stab:
            dir_hint, conf = self._detect_turn_direction(image)
            st_dir, st_conf = self.turn_stab.update(dir_hint, conf)
            st_conf = float(np.clip(st_conf, 0.0, 1.0))
            if st_dir is not None and st_conf > 0.0:
                base = self.turn_bias_gain.clamp(*self.config.turn_bias_bounds)
                extra = torch.tensor(
                    self.behavior["turn_bias_boost"], 
                    device=self.device,
                    dtype=steer_mean.dtype,
                )
                direction_sign = 1.0 if st_dir == 'right' else -1.0
                steer_mean = (
                    steer_mean + (base + extra) * st_conf * direction_sign
                ).clamp(-1.0, 1.0)

        # Behavioral scaling/bias on means
        steer_mean = (
            steer_mean * self.behavior["steer_gain"]
            + self.behavior["center_bias"]
        ).clamp(-1.0, 1.0)
        
        throttle_mean = (
            throttle_mean * self.behavior["throttle_gain"]
        ).clamp(0.0, 1.0)

        # Smoothness reduces stochasticity (1.0 -> near-deterministic).
        # We re-clamp to std_bounds after scaling to keep std strictly positive.
        std = self.log_std.exp().clamp(*self.config.std_bounds)
        smoothness = float(np.clip(self.behavior["smoothness"], 0.0, 1.0))
        std = (std * (1.0 - smoothness)).clamp(*self.config.std_bounds)
        dist_s = Normal(steer_mean, std[0])
        dist_t = Normal(throttle_mean, std[1])
    
        if explore:
            steer_s = dist_s.rsample()
            throt_s = dist_t.rsample()
            log_prob = dist_s.log_prob(steer_s) + dist_t.log_prob(throt_s)
            entropy = dist_s.entropy() + dist_t.entropy()
        else:
            steer_s, throt_s = steer_mean, throttle_mean
            log_prob = torch.zeros(1, device=self.device)
            entropy = torch.zeros(1, device=self.device)
    
        steering = float(steer_s.clamp(-1.0, 1.0).item())
        throttle = float(throt_s.clamp(0.0, 1.0).item())
        return steering, throttle, log_prob, entropy
    
    def _to_chw_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to canonical (C, H, W) float32 format.

        Accepts: (H, W) / (C, H, W) / (H, W, C)
        Returns: np.ndarray with shape (C, H, W), dtype float32.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")

        arr = image.astype(np.float32, copy=False)

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        elif arr.ndim == 3:
            expected_c = self.input_shape[0]

            # Already CHW
            if arr.shape[0] == expected_c:
                pass

            # HWC
            elif arr.shape[-1] == expected_c:
                arr = np.transpose(arr, (2, 0, 1))

            # Common RGB/BGR HWC case, even if expected_c == 1
            elif arr.shape[-1] in (1, 2, 3):
                arr = np.transpose(arr, (2, 0, 1))

            # Common CHW case
            elif arr.shape[0] in (1, 2, 3):
                pass

            else:
                raise ValueError(
                    f"Cannot infer channel axis for image shape {arr.shape}; "
                    f"expected input_shape={self.input_shape}"
                )

        else:
            raise ValueError(f"Unsupported image ndim={arr.ndim}")

        if arr.shape != self.input_shape:
            raise ValueError(f"Canonical image shape {arr.shape} != expected {self.input_shape}")

        return arr
    
    def _prepare_input(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor with shape (1, C, H, W)."""
        arr = self._to_chw_image(image)
        tensor = torch.from_numpy(arr).float().unsqueeze(0).to(self.device)
        return tensor
    
    def _to_edge_map_2d(self, image: np.ndarray) -> np.ndarray:
        """Convert input image to a 2D uint8 edge map with shape (H, W)."""
        chw = self._to_chw_image(image)

        if chw.shape[0] == 1:
            edge = chw[0]
        else:
            # Fallback for multi-channel input: convert to grayscale-like mean.
            edge = chw.mean(axis=0)

        if edge.max() <= 1.0:
            edge = edge * 255.0

        return edge.astype(np.uint8)


    def _detect_turn_direction(self, edge_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Lightweight left/right/None detector with confidence in [0, 1].

            Operates on an edge map of shape (1, 60, 80) or (60, 80).
            Combines two cues: pixel-density asymmetry between left/right halves
            of the upper ROI, and Hough-line evidence with angle/position gating.
        """
        try:
            img = self._to_edge_map_2d(edge_image)
            bin_img = (img > 127).astype(np.uint8)
            h, w = bin_img.shape
            diag_frac = getattr(self, "bp_diag_roi_row_start", 0.50)
            y_end = int(float(diag_frac) * h)

            bin_roi = bin_img[:y_end, :]
            img_roi = img[:y_end, :]
            hR, wR = bin_roi.shape
            if hR == 0 or wR == 0:
                return None, 0.0

            mid = bin_roi[int(0.15 * hR):int(0.55 * hR), :]
            low = bin_roi[int(0.40 * hR):, :]

            mid_left = int(mid[:, :wR // 2].sum())
            mid_right = int(mid[:, wR // 2:].sum())
            low_left = int(low[:, :wR // 2].sum())
            low_right = int(low[:, wR // 2:].sum())

            total_left = mid_left + low_left
            total_right = mid_right + low_right
            total = total_left + total_right
            if total < 0.005 * wR * hR:
                return None, 0.0

            asym = (total_right - total_left) / max(1, total)

            scale = max(1, int(0.02 * hR))
            lines = cv2.HoughLinesP(img_roi, 1, np.pi / 180, threshold=int(getattr(self, "bp_hough_threshold", 20)),
                                    minLineLength=5 * scale, maxLineGap=2 * scale)
            left_score = right_score = 0.0
            if lines is not None and len(lines) > 0:
                valid = 0
                for (x1, y1, x2, y2) in lines[:, 0, :]:
                    if x2 == x1:
                        continue
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    cx = 0.5 * (x1 + x2)
                    if abs(angle) < 8 or abs(angle) > 82:
                        continue
                    valid += 1
                    if cx < 0.4 * wR and 10 <= angle <= 55:
                        left_score += 1
                    elif cx > 0.6 * wR and -55 <= angle <= -10:
                        right_score += 1
                if valid > 0:
                    left_score /= valid
                    right_score /= valid

            left_conf = 0.45 * left_score + 0.55 * max(0, -asym)
            right_conf = 0.45 * right_score + 0.55 * max(0, asym)

            if left_conf > right_conf and left_conf > 0.3:
                return "left", float(min(left_conf, 1.0))
            if right_conf > 0.3:
                return "right", float(min(right_conf, 1.0))
            return None, 0.0

        except Exception as e:
            logger.debug("Turn detection failed: %s", e, exc_info=True)
            return None, 0.0

    # -------- Debug helpers --------
    def get_state_info(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "input_shape": self.input_shape,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.trainable_params()),
            "current_std": self.log_std.exp().detach().cpu().tolist(),
            "turn_bias_gain": float(self.turn_bias_gain.item()),
            "turn_stabilizer": {
                "prev_dir": self.turn_stab.prev_dir if self.turn_stab else None,
                "prev_conf": self.turn_stab.prev_conf if self.turn_stab else 0.0,
            } if self.turn_stab else None,
        }

    def reset_turn_stabilizer(self):
        if self.turn_stab:
            self.turn_stab.reset()

    def debug_weight_stats(self, top_k=10):
        rows = []
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                t = p.detach().float().cpu().view(-1)
                a = t.numpy()
                rows.append({
                    "name": name,
                    "shape": tuple(p.shape),
                    "n": int(a.size),
                    "min": float(np.min(a)),
                    "max": float(np.max(a)),
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a)),
                    "absmax": float(np.max(np.abs(a))),
                    "p99_abs": float(np.quantile(np.abs(a), 0.99)),
                })

        rows_sorted = sorted(rows, key=lambda r: r["absmax"], reverse=True)

        print("\n=== Weight stats (sorted by absmax) ===")
        for r in rows_sorted:
            print(f"{r['name']:<25} shape={r['shape']} "
                  f"min={r['min']:+.4f} max={r['max']:+.4f} mean={r['mean']:+.4f} std={r['std']:+.4f} "
                  f"absmax={r['absmax']:.4f} p99|.|={r['p99_abs']:.4f}")
 
        print("\n=== Top elements by |value| (global) ===")

        all_vals = []
        for name, p in self.model.named_parameters():
            t = p.detach().float().cpu().view(-1)
            a = t.numpy()

            idx = np.argpartition(np.abs(a), -min(top_k, a.size))[-min(top_k, a.size):]
            for i in idx:
                all_vals.append((float(abs(a[i])), float(a[i]), name))
        all_vals.sort(reverse=True, key=lambda x: x[0])
        for abs_v, v, name in all_vals[:top_k]:
            print(f"{name:<25} value={v:+.6f} |v|={abs_v:.6f}")

            