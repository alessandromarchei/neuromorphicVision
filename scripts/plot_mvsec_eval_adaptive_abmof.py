#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# =======================================================================
# PARSE RUN NAME
# =======================================================================
ABMOF_REGEX = re.compile(
    r"mvsec_(?P<scene>.+?)_P(?P<R>[\d.]+)",
    re.IGNORECASE
)

BASELINE_REGEX = re.compile(
    r"mvsec_(?P<scene>.+?)_dt(?P<dt>\d+)",
    re.IGNORECASE
)

# =======================================================================
# NEW: BEST CONFIG FOR INDOOR (3 SCENES) AND OUTDOOR
# =======================================================================
def select_best_indoor_outdoor(per_scene_adapt):
    """
    Indoor:
      - consider scenes whose name starts with 'indoor'
      - we require a config to:
          * appear in ALL indoor scenes
          * have dAEE < 0 in ALL indoor scenes (always improves)
      - among those, pick the one with the most negative average dAEE

    Outdoor:
      - for each outdoor scene (e.g. 'outdoor1')
      - pick config with most negative dAEE
    """
    indoor_scenes = [s for s in per_scene_adapt.keys() if s.lower().startswith("indoor")]
    outdoor_scenes = [s for s in per_scene_adapt.keys() if s.lower().startswith("outdoor")]

    # ---------------- INDOOR ----------------
    indoor_config_to_daeelist = defaultdict(list)  # cfg -> [(scene, dAEE), ...]

    for scene in indoor_scenes:
        for r in per_scene_adapt[scene]:
            if "dAEE" in r:
                indoor_config_to_daeelist[r["config_id"]].append((scene, r["dAEE"]))

    indoor_candidates = []
    for cfg, lst in indoor_config_to_daeelist.items():
        # must appear in ALL indoor scenes
        if len(lst) != len(indoor_scenes):
            continue
        dvals = [d for (_, d) in lst]
        # must improve in all scenes
        if not all(d < 0 for d in dvals):
            continue

        avg_dAEE = float(np.mean(dvals))
        max_dAEE = float(max(dvals))  # worst case (closest to 0)

        indoor_candidates.append({
            "config_id": cfg,
            "avg_dAEE": avg_dAEE,
            "max_dAEE": max_dAEE,
            "per_scene_dAEE": {scene: d for (scene, d) in lst},
        })

    best_indoor = None
    if indoor_candidates:
        # choose cfg with MOST negative average dAEE
        best_indoor = min(indoor_candidates, key=lambda x: x["avg_dAEE"])

    # ---------------- OUTDOOR ----------------
    best_outdoor_per_scene = {}
    for scene in outdoor_scenes:
        runs = [r for r in per_scene_adapt[scene] if "dAEE" in r]
        if not runs:
            continue
        # config with most negative dAEE in that outdoor scene
        best = min(runs, key=lambda x: x["dAEE"])
        best_outdoor_per_scene[scene] = {
            "config_id": best["config_id"],
            "dAEE": best["dAEE"],
            "dREE": best.get("dREE", np.nan),
        }

    return best_indoor, best_outdoor_per_scene



def parse_run_name(name):
    """
    Detect adaptive slicing (new scheme) or baseline dtX.
    """
    m = ABMOF_REGEX.match(name)
    if m:
        R = m.group("R")

        return {
            "scene": m.group("scene"),
            "type": "adaptive_abmof",
            "R": float(R),
            "config_id": f"R_{R}"
        }

    m = BASELINE_REGEX.match(name)
    if m:
        return {
            "scene": m.group("scene"),
            "type": "baseline",
            "config_id": f"dt{m.group('dt')}"
        }

    return None


# =======================================================================
# PARSE LOG FILE
# =======================================================================
def parse_logs(path):
    mean_aee = None
    median_aee = None
    mean_ree = None
    median_ree = None

    with open(path, "r") as f:
        for line in f:
            ln = line.strip()

            if ln.startswith("Mean AEE"):
                mean_aee = float(ln.split(":")[1])
            elif ln.startswith("Median AEE"):
                median_aee = float(ln.split(":")[1])
            elif ln.startswith("Mean REE"):
                mean_ree = float(ln.split(":")[1])
            elif ln.startswith("Median REE"):
                median_ree = float(ln.split(":")[1])

    if None in (mean_aee, median_aee, mean_ree, median_ree):
        return None

    return dict(
        mean_aee=mean_aee,
        median_aee=median_aee,
        mean_ree=mean_ree,
        median_ree=median_ree
    )


# =======================================================================
# SCAN RUNS
# =======================================================================
def analyze_runs(root_dir):
    per_scene_adapt = defaultdict(list)
    per_scene_baseline = defaultdict(list)
    all_configs = set()
    global_list = []

    dirs = os.listdir(root_dir)
    for idx, folder in enumerate(dirs):
        print(f"[INFO] Found run {idx+1}/{len(dirs)}: {folder}")

    for folder in os.listdir(root_dir):
        run_path = os.path.join(root_dir, folder)
        if not os.path.isdir(run_path):
            print(f"[WARN] Skipping non-directory: {run_path}, not a run folder.")
            continue

        parsed = parse_run_name(folder)
        if parsed is None:
            print(f"[WARN] Could not parse run name: {folder}")
            continue

        log_path = os.path.join(run_path, "logs.log")
        if not os.path.exists(log_path):
            print(f"[WARN] Missing logs.log in run folder: {run_path}")
            continue

        metrics = parse_logs(log_path)
        if metrics is None:
            print(f"[WARN] Could not parse logs.log in run folder: {run_path}")
            continue

        record = {
            "config_id": parsed["config_id"],
            "type": parsed["type"],
            **metrics
        }

        if parsed["type"] == "adaptive_abmof":
            record.update({
                "R": parsed["R"]
            })
            per_scene_adapt[parsed["scene"]].append(record)
            all_configs.add(record["config_id"])
        else:
            per_scene_baseline[parsed["scene"]].append(record)

        global_list.append({**record, "scene": parsed["scene"]})

    # ------------------------------------------------------------------
    # COMPUTE DELTAS FOR EACH SCENE
    # ------------------------------------------------------------------
    for scene in per_scene_adapt.keys():
        if scene not in per_scene_baseline:
            print(f"[WARN] No baseline found for scene {scene}")
            continue

        baseline = per_scene_baseline[scene][0]
        base_aee = baseline["mean_aee"]
        base_ree = baseline["median_ree"]

        for r in per_scene_adapt[scene]:
            r["dAEE"] = r["mean_aee"] - base_aee
            r["dREE"] = r["median_ree"] - base_ree

    return per_scene_adapt, per_scene_baseline, sorted(all_configs), global_list


# =======================================================================
# GLOBAL AGGREGATION
# =======================================================================
def compute_global(global_list, config_list):
    grouped = defaultdict(lambda:
                          {"mean_aee": [], "median_aee": [],
                           "mean_ree": [], "median_ree": []})

    for r in global_list:
        if r["type"] != "adaptive":
            continue
        cfg = r["config_id"]
        grouped[cfg]["mean_aee"].append(r["mean_aee"])
        grouped[cfg]["median_aee"].append(r["median_aee"])
        grouped[cfg]["mean_ree"].append(r["mean_ree"])
        grouped[cfg]["median_ree"].append(r["median_ree"])

    out = []
    for cfg in config_list:
        g = grouped[cfg]
        out.append({
            "config_id": cfg,
            "mean_aee": float(np.mean(g["mean_aee"])) if g["mean_aee"] else np.nan,
            "median_aee": float(np.median(g["median_aee"])) if g["median_aee"] else np.nan,
            "mean_ree": float(np.mean(g["mean_ree"])) if g["mean_ree"] else np.nan,
            "median_ree": float(np.median(g["median_ree"])) if g["median_ree"] else np.nan
        })

    return out


# =======================================================================
# PRINT SUMMARY TABLE
# =======================================================================
def print_summary(title, records):
    print(f"\n===================== {title} =====================")
    print(f"{'Config':>40} | {'AEE':>8} | {'REE':>8} | {'dAEE':>8} | {'dREE':>8}")
    print("-" * 100)

    for r in records:
        dAEE = r.get("dAEE", np.nan)
        dREE = r.get("dREE", np.nan)
        print(f"{r['config_id']:>40} | "
              f"{r['mean_aee']:8.4f} | "
              f"{r['median_ree']:8.4f} | "
              f"{dAEE:8.4f} | "
              f"{dREE:8.4f}")


# =======================================================================
# NEW: BEST CONFIG SELECTION PER SCENE
# =======================================================================
def select_best_configs(per_scene_adapt):
    """
    For each scene, select:
      - config with best (most negative) ΔAEE
      - config with best (most negative) ΔREE
      - config with best tradeoff (closest to origin in ΔAEE–ΔREE plane)
    """
    best_summary = {}

    for scene, runs in per_scene_adapt.items():
        # keep only runs that actually have dAEE/dREE
        valid = [r for r in runs if "dAEE" in r and "dREE" in r]
        if not valid:
            continue

        # Best ΔAEE (most negative dAEE)
        best_dAEE = min(valid, key=lambda x: x["dAEE"])

        # Best ΔREE (most negative dREE)
        best_dREE = min(valid, key=lambda x: x["dREE"])

        # Best tradeoff: minimize Euclidean distance from (0, 0)
        best_tradeoff = min(
            valid,
            key=lambda x: np.hypot(x["dAEE"], x["dREE"])
        )

        best_summary[scene] = {
            "best_dAEE": best_dAEE,
            "best_dREE": best_dREE,
            "best_tradeoff": best_tradeoff,
        }

    return best_summary


# =======================================================================
# PLOTTING
# =======================================================================
def plot_combined(per_scene_adapt, per_scene_baseline, global_row, config_list, out_dir):

    scenes = sorted(per_scene_adapt.keys())
    n_rows = 1 + len(scenes)

    fig, axes = plt.subplots(n_rows, 1, figsize=(22, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # ------------------ GLOBAL ROW ------------------
    ax1 = axes[0]
    ax2 = ax1.twinx()

    aee = [r["mean_aee"] for r in global_row]
    ree = [r["median_ree"] for r in global_row]

    ax1.plot(config_list, aee, marker="o", color="tab:blue")
    ax2.plot(config_list, ree, marker="x", color="tab:green")

    ax1.set_title("GLOBAL – Mean AEE vs Median REE")
    ax1.set_ylabel("AEE", color="tab:blue")
    ax2.set_ylabel("Median REE", color="tab:green")
    ax1.grid(True)

    # ------------------ SCENE ROWS ------------------
    for row_idx, scene in enumerate(scenes, start=1):
        ax1 = axes[row_idx]
        ax2 = ax1.twinx()

        lookup = {r["config_id"]: r for r in per_scene_adapt[scene]}

        aee_vals = [lookup[c]["mean_aee"] if c in lookup else np.nan for c in config_list]
        ree_vals = [lookup[c]["median_ree"] if c in lookup else np.nan for c in config_list]

        ax1.plot(config_list, aee_vals, marker="o", color="tab:blue")
        ax2.plot(config_list, ree_vals, marker="x", color="tab:green")

        # baseline
        if scene in per_scene_baseline:
            b = per_scene_baseline[scene][0]
            ax1.axhline(b["mean_aee"], linestyle="--", color="tab:blue", label="AEE dt=1")
            ax2.axhline(b["median_ree"], linestyle="--", color="tab:green", label="REE dt=1")

        ax1.set_title(f"{scene} – Mean AEE vs Median REE")
        ax1.set_ylabel("AEE", color="tab:blue")
        ax2.set_ylabel("Median REE", color="tab:green")
        ax1.grid(True)

        if scene in per_scene_baseline:
            ax1.legend(loc="upper left")

    # Only bottom subplot shows xticks
    plt.setp(axes[-1].get_xticklabels(), rotation=90)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "eval_adaptive.png"), dpi=250)

    print(f"[PLOT] Saved → {os.path.join(out_dir, 'eval_adaptive.png')}")


def plot_deltas(per_scene_adapt, config_list, out_dir):
    scenes = sorted(per_scene_adapt.keys())
    for scene in scenes:

        fig, ax1 = plt.subplots(figsize=(20, 6))
        ax2 = ax1.twinx()

        lookup = {r["config_id"]: r for r in per_scene_adapt[scene]}

        dAEE = [lookup[c]["dAEE"] if c in lookup else np.nan for c in config_list]
        dREE = [lookup[c]["dREE"] if c in lookup else np.nan for c in config_list]

        ax1.plot(config_list, dAEE, marker="o", color="tab:blue", label="ΔAEE")
        ax2.plot(config_list, dREE, marker="x", color="tab:red", label="ΔREE")

        ax1.axhline(0, linestyle="--", color="tab:blue")
        ax2.axhline(0, linestyle="--", color="tab:red")

        plt.setp(ax1.get_xticklabels(), rotation=90)

        ax1.set_title(f"{scene} – ΔAEE and ΔREE vs Baseline")
        ax1.set_ylabel("ΔAEE (lower = better)", color="tab:blue")
        ax2.set_ylabel("ΔREE (lower = better)", color="tab:red")
        ax1.grid(True)

        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{scene}_delta_plot.png"), dpi=200)
        plt.close()
        print(f"[PLOT] Saved delta plot → {scene}_delta_plot.png")


def plot_delta_scatter(per_scene_adapt, out_dir):
    for scene, runs in per_scene_adapt.items():

        fig = plt.figure(figsize=(8, 8))

        dAEE = [r["dAEE"] for r in runs]
        dREE = [r["dREE"] for r in runs]
        labels = [r["config_id"] for r in runs]

        plt.scatter(dAEE, dREE, c="tab:purple")

        # zero-lines
        plt.axvline(0, linestyle="--", color="black")
        plt.axhline(0, linestyle="--", color="black")

        # annotate points
        for x, y, lb in zip(dAEE, dREE, labels):
            plt.annotate(lb, (x, y), fontsize=6)

        plt.title(f"{scene} — ΔAEE vs ΔREE (Pareto View)")
        plt.xlabel("ΔAEE (negative = better)")
        plt.ylabel("ΔREE (negative = better)")
        plt.grid(True)

        path = os.path.join(out_dir, f"{scene}_scatter_deltas.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"[PLOT] Saved scatter → {path}")


# =======================================================================
# NEW: COOL PLOT HIGHLIGHTING BEST CONFIGS
# =======================================================================
def plot_best_configs(per_scene_adapt, best_summary, out_dir):
    """
    For each scene, make a ΔAEE–ΔREE scatter and highlight:
      - best ΔAEE (red star)
      - best ΔREE (green star)
      - best tradeoff (blue diamond)
    """
    os.makedirs(out_dir, exist_ok=True)

    for scene, runs in per_scene_adapt.items():
        if scene not in best_summary:
            continue

        fig = plt.figure(figsize=(8, 8))
        dAEE = [r["dAEE"] for r in runs]
        dREE = [r["dREE"] for r in runs]
        labels = [r["config_id"] for r in runs]

        # all points
        plt.scatter(dAEE, dREE, s=50, alpha=0.5, label="All configs")

        # winners
        bA = best_summary[scene]["best_dAEE"]
        bR = best_summary[scene]["best_dREE"]
        bT = best_summary[scene]["best_tradeoff"]

        plt.scatter([bA["dAEE"]], [bA["dREE"]], s=200, marker="*", color="red", label="Best ΔAEE")
        plt.scatter([bR["dAEE"]], [bR["dREE"]], s=200, marker="*", color="green", label="Best ΔREE")
        plt.scatter([bT["dAEE"]], [bT["dREE"]], s=160, marker="D", color="blue", label="Best tradeoff")

        # zero-lines
        plt.axvline(0, linestyle="--", color="black", linewidth=1)
        plt.axhline(0, linestyle="--", color="black", linewidth=1)

        # annotate all configs (small font)
        for x, y, lb in zip(dAEE, dREE, labels):
            plt.annotate(lb, (x, y), fontsize=6, alpha=0.8)

        plt.title(f"{scene} — Best Configs (ΔAEE vs ΔREE)")
        plt.xlabel("ΔAEE (negative = better)")
        plt.ylabel("ΔREE (negative = better)")
        plt.grid(True, alpha=0.4)
        plt.legend(loc="best")

        path = os.path.join(out_dir, f"{scene}_best_configs.png")
        plt.savefig(path, dpi=250)
        plt.close()
        print(f"[PLOT] Saved best-config plot → {path}")


# =======================================================================
# MAIN
# =======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="results/runs_mvsec_adaptive")
    parser.add_argument("--out_dir", type=str, default="analysis_plots_adaptive_mvsec")
    args = parser.parse_args()

    per_scene_adapt, per_scene_baseline, config_list, global_list = analyze_runs(args.runs_dir)

    # global aggregation
    global_row = compute_global(global_list, config_list)

    print_summary("GLOBAL SUMMARY", global_row)

    for scene in sorted(per_scene_adapt.keys()):
        sorted_scene = sorted(per_scene_adapt[scene], key=lambda x: x["config_id"])
        print_summary(f"SCENE {scene.upper()}", sorted_scene)

        # ------------------------------------------------------------------
    # NEW: select best configs per scene
    # ------------------------------------------------------------------
    best_summary = select_best_configs(per_scene_adapt)

    print("\n===================== BEST CONFIGS PER SCENE =====================")
    for scene in sorted(best_summary.keys()):
        b = best_summary[scene]
        print(f"\n[SCENE] {scene}")
        print(f"  -> Best ΔAEE      : {b['best_dAEE']['config_id']}  "
              f"dAEE={b['best_dAEE']['dAEE']:.4f}, dREE={b['best_dAEE']['dREE']:.4f}")
        print(f"  -> Best ΔREE      : {b['best_dREE']['config_id']}  "
              f"dAEE={b['best_dREE']['dAEE']:.4f}, dREE={b['best_dREE']['dREE']:.4f}")
        print(f"  -> Best tradeoff  : {b['best_tradeoff']['config_id']}  "
              f"dAEE={b['best_tradeoff']['dAEE']:.4f}, dREE={b['best_tradeoff']['dREE']:.4f}")

    # ------------------------------------------------------------------
    # NEW: best config for INDOOR (3 scenes) and OUTDOOR
    # ------------------------------------------------------------------
    best_indoor, best_outdoor = select_best_indoor_outdoor(per_scene_adapt)

    print("\n===================== GLOBAL BEST FOR INDOOR / OUTDOOR =====================")

    if best_indoor is not None:
        print("\n[INDOOR] Best config that improves AEE in ALL indoor scenes:")
        print(f"  Config ID: {best_indoor['config_id']}")
        print(f"  Avg dAEE over indoor scenes: {best_indoor['avg_dAEE']:.4f} (more negative = better)")
        print(f"  Worst dAEE over indoor scenes: {best_indoor['max_dAEE']:.4f}")
        print("  Per-scene dAEE:")
        for scene, d in best_indoor["per_scene_dAEE"].items():
            print(f"    - {scene}: dAEE={d:.4f}")
    else:
        print("\n[INDOOR] No single config improves AEE in ALL indoor scenes with dAEE < 0.")

    for scene, info in best_outdoor.items():
        print(f"\n[OUTDOOR] Scene {scene}: best config by dAEE:")
        print(f"  Config ID: {info['config_id']}")
        print(f"  dAEE={info['dAEE']:.4f}, dREE={info['dREE']:.4f}")

    # existing plots
    plot_combined(per_scene_adapt, per_scene_baseline, global_row, config_list, args.out_dir)
    plot_deltas(per_scene_adapt, config_list, args.out_dir)
    plot_delta_scatter(per_scene_adapt, args.out_dir)

    # NEW: cool Pareto plots with best highlighted
    plot_best_configs(per_scene_adapt, best_summary, args.out_dir)


if __name__ == "__main__":
    main()
