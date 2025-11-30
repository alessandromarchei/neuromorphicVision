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
    r"mvsec_(?P<scene>.+?)_P(?P<P>[\d.]+)_I(?P<I>[\d.]+)_sp(?P<sp>[\d.]+)_th(?P<th>[\d.]+)_step(?P<step>\d+)",
    re.IGNORECASE
)

BASELINE_REGEX = re.compile(
    r"mvsec_(?P<scene>.+?)_dt(?P<dt>\d+)",
    re.IGNORECASE
)

def parse_run_name(name):
    """
    Detect adaptive slicing (new scheme) or baseline dtX.
    """
    m = PID_REGEX.match(name)
    if m:
        P = m.group("P")
        I = m.group("I")
        sp = m.group("sp")
        th = m.group("th")
        step = m.group("step")

        return {
            "scene": m.group("scene"),
            "type": "adaptive",
            "P": float(P),
            "I": float(I),
            "SP": float(sp),
            "TH": float(th),
            "STEP": int(step),
            "config_id": f"P{P}_I{I}_SP{sp}_T{th}_STEP{step}"
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
            record.update({
                "P": parsed["P"],
                "I": parsed["I"],
                "SP": parsed["SP"],
                "TH": parsed["TH"],
                "STEP": parsed["STEP"],
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
            "mean_aee": float(np.mean(g["mean_aee"])),
            "median_aee": float(np.median(g["median_aee"])),
            "mean_ree": float(np.mean(g["mean_ree"])),
            "median_ree": float(np.median(g["median_ree"]))
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
            ax1.axhline(b["mean_aee"], linestyle="--", color="tab:blue", label="Baseline AEE")
            ax2.axhline(b["median_ree"], linestyle="--", color="tab:green", label="Baseline REE")

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

        ax1.axhline(0, linestyle="--", color="black")
        ax2.axhline(0, linestyle="--", color="black")

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

    print_summary("GLOBAL SUMMARY", global_row)

    for scene in sorted(per_scene_adapt.keys()):
        sorted_scene = sorted(per_scene_adapt[scene], key=lambda x: x["config_id"])
        print_summary(f"SCENE {scene.upper()}", sorted_scene)

    plot_combined(per_scene_adapt, per_scene_baseline, global_row, config_list, args.out_dir)

    #adding the plot delta and plot scatter functions
    plot_deltas(per_scene_adapt, config_list, args.out_dir)
    plot_delta_scatter(per_scene_adapt, args.out_dir)



if __name__ == "__main__":
    main()
