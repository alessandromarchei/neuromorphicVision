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
PID_REGEX = re.compile(
    r"mvsec_(?P<scene>.+?)_P(?P<P>[\d.]+)_I(?P<I>[\d.]+)_sp(?P<sp>\d+)_th(?P<th>\d+)",
    re.IGNORECASE
)

BASELINE_REGEX = re.compile(
    r"mvsec_(?P<scene>.+?)_dt(?P<dt>\d+)",
    re.IGNORECASE
)


def parse_run_name(name):
    m = PID_REGEX.match(name)
    if m:
        return {
            "scene": m.group("scene"),
            "type": "adaptive",
            "config_id": f"P{m.group('P')}_I{m.group('I')}_SP{m.group('sp')}_T{m.group('th')}"
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

    for folder in os.listdir(root_dir):
        run_path = os.path.join(root_dir, folder)
        if not os.path.isdir(run_path):
            continue

        parsed = parse_run_name(folder)
        if parsed is None:
            continue

        log_path = os.path.join(run_path, "logs.log")
        if not os.path.exists(log_path):
            continue

        metrics = parse_logs(log_path)
        if metrics is None:
            continue

        record = {
            "config_id": parsed["config_id"],
            "type": parsed["type"],
            **metrics
        }

        if parsed["type"] == "adaptive":
            per_scene_adapt[parsed["scene"]].append(record)
            all_configs.add(parsed["config_id"])
        else:
            per_scene_baseline[parsed["scene"]].append(record)

        global_list.append({**record, "scene": parsed["scene"]})

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
            "mean_aee": float(np.mean(g["mean_aee"])),
            "median_aee": float(np.median(g["median_aee"])),
            "mean_ree": float(np.mean(g["mean_ree"])),
            "median_ree": float(np.median(g["median_ree"]))
        })

    return out


# =======================================================================
# PRINT SUMMARY LIKE ORIGINAL SCRIPT
# =======================================================================
def print_summary(title, records):
    print(f"\n===================== {title} =====================")
    print(f"{'Config':>28} | {'AEE_mean':>8} | {'AEE_med':>8} | {'REE_mean':>8} | {'REE_med':>8}")
    print("-" * 80)

    for r in records:
        print(f"{r['config_id']:>28} | "
              f"{r['mean_aee']:8.4f} | "
              f"{r['median_aee']:8.4f} | "
              f"{r['mean_ree']:8.4f} | "
              f"{r['median_ree']:8.4f}")


# =======================================================================
# PLOTTING
# =======================================================================
def plot_combined(per_scene_adapt, per_scene_baseline, global_row, config_list, out_dir):

    scenes = sorted(per_scene_adapt.keys())
    n_rows = 1 + len(scenes)

    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 4 * n_rows), sharex=True)

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

        # adaptive lookup
        lookup = {r["config_id"]: r for r in per_scene_adapt[scene]}
        aee_vals = [lookup[c]["mean_aee"] if c in lookup else np.nan for c in config_list]
        ree_vals = [lookup[c]["median_ree"] if c in lookup else np.nan for c in config_list]

        ax1.plot(config_list, aee_vals, marker="o", color="tab:blue")
        ax2.plot(config_list, ree_vals, marker="x", color="tab:green")

        # --- baseline row ---
        if scene in per_scene_baseline:
            b = per_scene_baseline[scene][0]
            ax1.axhline(b["mean_aee"], linestyle="--", color="tab:red", label="Baseline AEE")
            ax2.axhline(b["median_ree"], linestyle="--", color="tab:orange", label="Baseline REE")

        ax1.set_title(f"{scene} – Mean AEE vs Median REE")
        ax1.set_ylabel("AEE", color="tab:blue")
        ax2.set_ylabel("Median REE", color="tab:green")
        ax1.grid(True)

        if scene in per_scene_baseline:
            ax1.legend(loc="upper left")

    # Only bottom subplot prints xticks
    plt.setp(axes[-1].get_xticklabels(), rotation=90)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "combined_pid_with_baseline.png"), dpi=250)

    print(f"[PLOT] Saved → {os.path.join(out_dir, 'combined_pid_with_baseline.png')}")


# =======================================================================
# MAIN
# =======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs_mvsec_adaptive")
    parser.add_argument("--out_dir", type=str, default="analysis_plots_pid_config")
    args = parser.parse_args()

    per_scene_adapt, per_scene_baseline, config_list, global_list = analyze_runs(args.runs_dir)

    # global aggregation
    global_row = compute_global(global_list, config_list)

    # print summaries
    print_summary("GLOBAL SUMMARY", global_row)

    for scene in sorted(per_scene_adapt.keys()):
        print_summary(f"SCENE {scene.upper()}", sorted(per_scene_adapt[scene], key=lambda x: x["config_id"]))

    # plotting
    plot_combined(per_scene_adapt, per_scene_baseline, global_row, config_list, args.out_dir)


if __name__ == "__main__":
    main()
