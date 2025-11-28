#!/usr/bin/env python3
import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Helpers to parse run folder name and logs.log
# ---------------------------------------------------------

RUN_NAME_REGEX = re.compile(
    r"mvsec_(?P<scene>.+?)_feat(?P<feat>\d+)_mag(?P<mag>\d+)", re.IGNORECASE
)

def parse_run_name(run_name):
    m = RUN_NAME_REGEX.match(run_name)
    if not m:
        return None
    scene = m.group("scene")
    feat = int(m.group("feat"))
    mag = int(m.group("mag"))
    return scene, feat, mag


def parse_logs_file(log_path):
    mean_aee = None
    median_aee = None
    mean_points = None
    mean_outliers = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Mean AEE"):
                try: mean_aee = float(line.split(":")[-1])
                except: pass

            elif line.startswith("Median AEE"):
                try: median_aee = float(line.split(":")[-1])
                except: pass

            elif line.startswith("Mean #points/frame"):
                try: mean_points = float(line.split(":")[-1])
                except: pass

            elif line.startswith("Mean Outliers"):
                try:
                    # line looks like: "Mean Outliers (%)    : 4.30"
                    mean_outliers = float(line.split(":")[-1])
                except:
                    pass

    if (
        mean_aee is None or
        median_aee is None or
        mean_points is None or
        mean_outliers is None
    ):
        print(f"[WARN] Missing some metrics in {log_path}")
        return None

    return mean_aee, median_aee, mean_points, mean_outliers



# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

def analyze_runs(root_dir):
    """
    Returns:
        records: flat list of runs
        grouped_global: dict[(feat, mag)] -> aggregated across ALL scenes
        grouped_by_scene: dict[scene][(feat,mag)] -> aggregated per scene
    """

    records = []

    # 1) Scan all folders
    for entry in os.listdir(root_dir):
        run_path = os.path.join(root_dir, entry)
        if not os.path.isdir(run_path):
            continue

        parsed = parse_run_name(entry)
        if parsed is None:
            continue

        scene, feat, mag = parsed
        log_path = os.path.join(run_path, "logs.log")
        if not os.path.isfile(log_path):
            continue

        metrics = parse_logs_file(log_path)
        if metrics is None:
            continue

        mean_aee, median_aee, mean_points, mean_outliers = metrics

        records.append(
            dict(run_name=entry, scene=scene, feat=feat, mag=mag,
                 mean_aee=mean_aee, median_aee=median_aee, mean_points=mean_points, mean_outliers=mean_outliers)
        )

    if not records:
        print("[ERROR] No valid runs found.")
        return [], {}, {}

    # ---------------------------------------------------------
    # 2) GLOBAL aggregation: group by (feat, mag)
    # ---------------------------------------------------------
    gdict = defaultdict(list)
    for r in records:
        gdict[(r["feat"], r["mag"])].append(r)

    grouped_global = {}
    for (feat, mag), lst in gdict.items():
        meanA = np.array([x["mean_aee"] for x in lst])
        medA = np.array([x["median_aee"] for x in lst])
        pts  = np.array([x["mean_points"] for x in lst])
        outs = np.array([x["mean_outliers"] for x in lst])

        grouped_global[(feat, mag)] = dict(
            feat=feat,
            mag=mag,
            n_runs=len(lst),
            avg_mean_aee=float(meanA.mean()),
            std_mean_aee=float(meanA.std()),
            avg_median_aee=float(medA.mean()),
            std_median_aee=float(medA.std()),
            avg_mean_points=float(pts.mean()),
            std_mean_points=float(pts.std()),
            avg_mean_outliers=float(outs.mean()),
            std_mean_outliers=float(outs.std()),
        )

    # ---------------------------------------------------------
    # 3) PER-SCENE aggregation
    # ---------------------------------------------------------
    grouped_by_scene = defaultdict(lambda: defaultdict(list))

    for r in records:
        grouped_by_scene[r["scene"]][(r["feat"], r["mag"])].append(r)

    per_scene_final = {}

    for scene, combos in grouped_by_scene.items():
        per_scene_final[scene] = {}
        for (feat, mag), lst in combos.items():

            meanA = np.array([x["mean_aee"] for x in lst])
            medA  = np.array([x["median_aee"] for x in lst])
            pts   = np.array([x["mean_points"] for x in lst])
            outs = np.array([x["mean_outliers"] for x in lst])

            per_scene_final[scene][(feat, mag)] = dict(
                feat=feat,
                mag=mag,
                n_runs=len(lst),
                avg_mean_aee=float(meanA.mean()),
                avg_median_aee=float(medA.mean()),
                avg_mean_points=float(pts.mean()),
                avg_mean_outliers=float(outs.mean())
            )

    return records, grouped_global, per_scene_final


# ---------------------------------------------------------
# Text printing
# ---------------------------------------------------------

def print_summary(title, grouped):
    print(f"\n================= {title} =================")
    print(
        f"{'feat':>6} {'mag':>6} {'n':>4} | "
        f"{'meanAEE':>8} | {'medAEE':>8} | {'#pts':>8} | {'out(%)':>8}"
    )

    print("-" * 60)

    for key in sorted(grouped.keys()):
        g = grouped[key]
        print(f"{g['feat']:6d} {g['mag']:6d} {g['n_runs']:4d} | "
            f"{g['avg_mean_aee']:8.4f} | "
            f"{g['avg_median_aee']:8.4f} | "
            f"{g['avg_mean_points']:8.2f} | "
            f"{g.get('avg_mean_outliers', 0.0):8.2f}")


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_config_row(ax_mean, ax_median, ax_points, grouped, title_prefix):
    """
    Plot a single row (3 subplots) for one configuration summary.
    """
    if not grouped:
        return

    feats = sorted(set(k[0] for k in grouped.keys()))
    mags = sorted(set(k[1] for k in grouped.keys()))
    cmap = plt.get_cmap("tab10")

    for idx, mag in enumerate(mags):
        color = cmap(idx % 10)

        xs = []
        ys_mean = []
        ys_median = []
        ys_points = []

        for feat in feats:
            key = (feat, mag)
            if key not in grouped:
                continue

            g = grouped[key]
            xs.append(feat)
            ys_mean.append(g["avg_mean_aee"])
            ys_median.append(g["avg_median_aee"])
            ys_points.append(g["avg_mean_points"])

        if xs:
            xs = np.array(xs)
            ax_mean.plot(xs, ys_mean, marker="o", label=f"mag={mag}", color=color)
            ax_median.plot(xs, ys_median, marker="o", label=f"mag={mag}", color=color)
            ax_points.plot(xs, ys_points, marker="o", label=f"mag={mag}", color=color)

    # Titles & labels
    ax_mean.set_title(f"{title_prefix} – Mean AEE")
    ax_mean.set_xlabel("desiredFeatures")
    ax_mean.grid(True)

    ax_median.set_title(f"{title_prefix} – Median AEE")
    ax_median.set_xlabel("desiredFeatures")
    ax_median.grid(True)

    ax_points.set_title(f"{title_prefix} – Mean #points")
    ax_points.set_xlabel("desiredFeatures")
    ax_points.grid(True)

    # Legend only on mean subplot to save space
    ax_mean.legend(loc="upper right")


def plot_all_summaries(global_grouped, grouped_by_scene, out_path):
    """
    Create one big figure:

       ROW 1 → GLOBAL summary
       ROW 2 → indoor1
       ROW 3 → indoor2
       ROW 4 → indoor3
       ROW 5 → outdoor1
       ...
    """
    scenes = sorted(grouped_by_scene.keys())  # indoor1, indoor2, ...
    n_rows = 1 + len(scenes)

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(18, 5 * n_rows),
        squeeze=False
    )

    # ---- FIRST ROW = GLOBAL SUMMARY ----
    print("[PLOT] Drawing GLOBAL summary...")
    plot_config_row(
        axes[0, 0], axes[0, 1], axes[0, 2],
        global_grouped,
        title_prefix="Global"
    )

    # ---- PER-SCENE ROWS ----
    for i, scene in enumerate(scenes):
        print(f"[PLOT] Drawing scene summary: {scene}")
        grouped_scene = grouped_by_scene[scene]
        row = i + 1  # because 0 is global

        plot_config_row(
            axes[row, 0], axes[row, 1], axes[row, 2],
            grouped_scene,
            title_prefix=f"Scene {scene}"
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved combined plot to: {out_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--out_dir",  type=str, default="analysis_plots")
    args = parser.parse_args()

    records, grouped_global, grouped_by_scene = analyze_runs(args.runs_dir)

    if not grouped_global:
        print("No valid data.")
        return

    # ========== PRINT GLOBAL SUMMARY ==========
    print_summary("GLOBAL CONFIG SUMMARY", grouped_global)

    # ========== PRINT PER-SCENE SUMMARY ==========
    for scene, grouped_scene in grouped_by_scene.items():
        print_summary(f"SCENE {scene.upper()}", grouped_scene)

    plot_all_summaries(
        global_grouped=grouped_global,
        grouped_by_scene=grouped_by_scene,
        out_path=os.path.join(args.out_dir, "combined_summary.png")
    )


if __name__ == "__main__":
    main()
