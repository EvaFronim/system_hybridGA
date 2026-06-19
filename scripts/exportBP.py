
import argparse, json, math, sys
import pandas as pd

# Behavior parameter keys as used in your GA stages
PARAM_KEYS = [
    "lane_kp_off","lane_kp_head","lane_kp_curv","curv_thr_slow",
    "lane_conf_thr","canny_scale","hough_threshold","diag_roi_row_start"
]

def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return x

def extract_bp(obj):
    """
    Accepts either:
      - flat dict with BP keys
      - dict with "behavior_params": {...}
      - dict with nested under "params" or "bp"
    Returns dict with only the BP keys if found.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    # try common nests
    candidates = []
    if isinstance(obj, dict):
        candidates.append(obj)
        if "behavior_params" in obj and isinstance(obj["behavior_params"], dict):
            candidates.append(obj["behavior_params"])
        if "bp" in obj and isinstance(obj["bp"], dict):
            candidates.append(obj["bp"])
        if "params" in obj and isinstance(obj["params"], dict):
            candidates.append(obj["params"])
    for cand in candidates:
        keys = set(cand.keys())
        if any(k in keys for k in PARAM_KEYS):
            out = {}
            for k in PARAM_KEYS:
                if k in cand:
                    out[k] = _coerce_float(cand[k])
            # post-process integer field
            if "hough_threshold" in out:
                try:
                    out["hough_threshold"] = int(round(float(out["hough_threshold"])))
                except Exception:
                    pass
            return out
    return None

def pick_row(df, mode="fitness"):
    if df.empty:
        raise SystemExit("Empty trials.csv")
    if mode == "fitness":
        idx = df["fitness"].astype(float).idxmax()
        return df.loc[idx]
    elif mode == "best_so_far":
        # pick the last row (highest trial index), which carries the final best_so_far
        return df.iloc[-1]
    else:
        raise ValueError("mode must be 'fitness' or 'best_so_far'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True, help="Path to trials.csv")
    ap.add_argument("--out", default="fixed_bp.json", help="Output JSON path")
    ap.add_argument("--mode", default="fitness", choices=["fitness","best_so_far"],
                    help="Pick row by max fitness or final best_so_far row")
    args = ap.parse_args()

    df = pd.read_csv(args.trials)
    if "params_json" not in df.columns:
        raise SystemExit("trials.csv has no 'params_json' column.")
    row = pick_row(df, mode=args.mode)
    params_raw = row["params_json"]
    bp = extract_bp(params_raw)
    if not bp:
        # try to parse entire column until we find a row with BP
        for v in df["params_json"]:
            bp = extract_bp(v)
            if bp:
                break
    if not bp:
        raise SystemExit("Could not locate behavior params in params_json.")

    with open(args.out, "w") as f:
        json.dump(bp, f, indent=2)
    print(f"[OK] Wrote {args.out}")
    print(json.dumps(bp, indent=2))

if __name__ == "__main__":
    main()
