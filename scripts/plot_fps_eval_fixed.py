#!/usr/bin/env python3
import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------
# REGEX to detect run naming scheme
# Example convention assumed:
# fpv_sceneName_paramK_paramB_paramC
# Modify accordingly if you have different naming
# ----------------------------------------------

RUN_NAME_REGEX = re.compile(
    r"fpv_(?P<scene>[a-zA-Z0-9_]+)_dt(?P<dt>[0-9]+)_c(?P<cval>[0-9.]+)",
    re.IGNORECASE
)

def parse_run_name(run_name):
    """
    Expected run name format:
        fpv_<scene>_dt<window_ms>_c<comp_filter_gain>

    Example:
        fpv_indoor_45_9_dt30_c4.0
    """
    m = RUN_NAME_REGEX.match(run_name)
    if not m:
        print(f"[WARN] Could not parse run name: {run_name}")
        return None

    scene   = m.group("scene")              # e.g., indoor_45_9
    dt_ms   = int(m.group("dt"))            # e.g., 30 → 30ms window
    c_gain  = float(m.group("cval"))         # e.g., 4.0

    # configuration label
    cfg = f"dt{dt_ms}_c{c_gain}"

    return scene, dt_ms, c_gain, cfg



# ----------------------------------------------
# Parse log files for altitude metrics
# ----------------------------------------------

def parse_altitude_log(log_path):
    """
    Parses FPV altitude result logs formatted like:

    Errore Assoluto Medio (MAE) sull'Altitudine: 0.4845 m
    Errore Relativo Medio (MRE) sull'Altitudine: 46.10 %

    Returns:
        mean_abs_error, rel_error_percent
        or None if missing
    """
    mae = None
    mre = None

    # Regex robusto per catturare numeri float
    mae_re = re.compile(r"Errore Assoluto Medio.*:\s*([0-9.,]+)")
    mre_re = re.compile(r"Errore Relativo Medio.*:\s*([0-9.,]+)")

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # match MAE
            m1 = mae_re.search(line)
            if m1:
                try:
                    mae = float(m1.group(1).replace(",", "."))
                except:
                    pass

            # match MRE (%)
            m2 = mre_re.search(line)
            if m2:
                try:
                    mre = float(m2.group(1).replace(",", "."))
                except:
                    pass

    if mae is None or mre is None:
        print(f"[WARN] Could not extract metrics from {log_path}")
        return None

    return mae, mre

# ----------------------------------------------
# Main scanning
# ----------------------------------------------

def analyze_runs_altitude(root_dir):
    records = []

    # Scan folders
    for entry in os.listdir(root_dir):
        run_path = os.path.join(root_dir, entry)
        if not os.path.isdir(run_path):
            continue

        parsed = parse_run_name(entry)
        if parsed is None:
            continue

        scene, dt, cgain, cfg = parsed

        log_path = os.path.join(run_path, "results.log")
        if not os.path.isfile(log_path):
            continue

        metrics = parse_altitude_log(log_path)

        if metrics is None:
            continue

        mean_alt, rel_alt = metrics

        records.append(dict(
            run_name   = entry,
            scene      = scene,
            dt_ms      = dt,
            comp_gain  = cgain,
            cfg        = cfg,
            mean_altitude = mean_alt,
            rel_altitude  = rel_alt
        ))

    if not records:
        print("[ERROR] No valid FPV altitude runs found.")
        return [], {}, {}

    # ---------------------------------------------------------
    # GROUPING BY CONFIG (GLOBAL)
    # ---------------------------------------------------------
    gdict = defaultdict(list)
    for r in records:
        gdict[r["cfg"]].append(r)

    grouped_global = {}
    for cfg, lst in gdict.items():
        aa = np.array([x["mean_altitude"] for x in lst])
        rr = np.array([x["rel_altitude"] for x in lst])

        grouped_global[cfg] = dict(
            cfg=cfg,
            n_runs=len(lst),
            mean_error=float(aa.mean()),
            rel_error=float(rr.mean()),
            std_abs=float(aa.std()),
            std_rel=float(rr.std())
        )

    # ---------------------------------------------------------
    # PER-SCENE
    # ---------------------------------------------------------
    scene_dict = defaultdict(lambda: defaultdict(list))

    for r in records:
        scene_dict[r["scene"]][r["cfg"]].append(r)

    grouped_by_scene = {}
    for scene, configs in scene_dict.items():
        grouped_by_scene[scene] = {}
        for cfg, lst in configs.items():
            aa = np.array([x["mean_altitude"] for x in lst])
            rr = np.array([x["rel_altitude"] for x in lst])

            grouped_by_scene[scene][cfg] = dict(
                cfg=cfg,
                mean_error=float(aa.mean()),
                rel_error=float(rr.mean()),
                n_runs=len(lst)
            )

    return records, grouped_global, grouped_by_scene


# ----------------------------------------------
# Plotting utilities
# ----------------------------------------------

def plot_scene_summary(ax_abs, ax_rel, grouped_cfg, title):
    if not grouped_cfg:
        return

    cfgs = sorted(grouped_cfg.keys())
    abs_vals = [grouped_cfg[c]["mean_error"] for c in cfgs]
    rel_vals = [grouped_cfg[c]["rel_error"] for c in cfgs]

    xs = np.arange(len(cfgs))

    ax_abs.plot(xs, abs_vals, marker="o")
    ax_abs.set_title(f"{title} – Mean Altitude Error (m)")
    ax_abs.set_xticks(xs)
    ax_abs.set_xticklabels(cfgs, rotation=45, ha="right")
    ax_abs.grid(True)

    ax_rel.plot(xs, rel_vals, marker="o", color="tab:red")
    ax_rel.set_title(f"{title} – Relative Altitude Error (%)")
    ax_rel.set_xticks(xs)
    ax_rel.set_xticklabels(cfgs, rotation=45, ha="right")
    ax_rel.grid(True)


def plot_all_altitude_summaries(grouped_global, grouped_by_scene, out_path):
    scenes = sorted(grouped_by_scene.keys())
    nrows = 1 + len(scenes)  # global + per-scene

    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(20, 6 * nrows),
        squeeze=False
    )

    # Global
    print("[PLOT] Building GLOBAL plot...")
    plot_scene_summary(
        axes[0, 0], axes[0, 1],
        grouped_global,
        title="Global"
    )

    # Per scene
    for i, scene in enumerate(scenes):
        print(f"[PLOT] Building scene plot: {scene}")
        plot_scene_summary(
            axes[i + 1, 0], axes[i + 1, 1],
            grouped_by_scene[scene],
            title=f"Scene {scene}"
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[PLOT] Saved altitude summary to {out_path}")


# ----------------------------------------------
# CLI
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs_fpv_fixed/")
    parser.add_argument("--out_dir", type=str, default="analysis_altitude_fpv_fixed/")
    args = parser.parse_args()

    records, grouped_global, grouped_by_scene = analyze_runs_altitude(args.runs_dir)

    if not grouped_global:
        print("No valid altitude data found.")
        return

    print("\n========= GLOBAL ALTITUDE SUMMARY =========")
    for cfg, g in grouped_global.items():
        print(f"{cfg:20s} | mean={g['mean_error']:.3f} | rel={g['rel_error']:.2f}%")

    for scene, grouped in grouped_by_scene.items():
        print(f"\n========= SCENE: {scene.upper()} =========")
        for cfg, g in grouped.items():
            print(f"{cfg:15s} | mean={g['mean_error']:.3f} | rel={g['rel_error']:.2f}%")

    output_path = os.path.join(args.out_dir, "fpv_altitude_summary.png")
    plot_all_altitude_summaries(
        grouped_global, grouped_by_scene,
        out_path=output_path
    )


if __name__ == "__main__":
    main()
