#!/usr/bin/env python3
import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# REGEX per nomi run
# ============================================================

# Fixed slicing:
#   fpv_indoor_45_2_dt30_c3.0
FIXED_REGEX = re.compile(
    r"fpv_(?P<scene>.+?)_dt(?P<dt>\d+)_c(?P<c>[\d.]+)$",
    re.IGNORECASE
)

# Adaptive slicing:
#   fpv_indoor_45_2_P0.3_I0.05_sp3_th5
ADAPT_REGEX = re.compile(
    r"fpv_(?P<scene>.+?)_P(?P<P>[\d.]+)_I(?P<I>[\d.]+)_sp(?P<sp>[\d.]+)_th(?P<th>[\d.]+)$",
    re.IGNORECASE
)


# ============================================================
# PARSING NOMI RUN
# ============================================================

def parse_fixed_name(name):
    m = FIXED_REGEX.match(name)
    if not m:
        return None
    scene = m.group("scene")
    dt = int(m.group("dt"))
    c = float(m.group("c"))
    config_id = f"dt{dt}_c{c}"
    return dict(
        scene=scene,
        type="fixed",
        dt=dt,
        C=c,
        config_id=config_id,
    )


def parse_adapt_name(name):
    m = ADAPT_REGEX.match(name)
    if not m:
        return None
    scene = m.group("scene")
    P = float(m.group("P"))
    I = float(m.group("I"))
    sp = float(m.group("sp"))
    th = float(m.group("th"))
    # P e I sono costanti ma li teniamo comunque nel record
    config_id = f"sp{sp}_th{th}"
    return dict(
        scene=scene,
        type="adaptive",
        P=P,
        I=I,
        SP=sp,
        TH=th,
        config_id=config_id,
    )


# ============================================================
# PARSING results.log
# ============================================================

def parse_results_log(log_path):
    """
    Estrarre:
      - MAE (in metri)
      - MRE (in percentuale, senza il simbolo %)
    """
    mae = None
    mre = None

    with open(log_path, "r") as f:
        for line in f:
            ln = line.strip()

            if "Errore Assoluto Medio (MAE)" in ln:
                # es: "Errore Assoluto Medio (MAE) sull'Altitudine: 0.4845 m"
                try:
                    val_str = ln.split(":")[-1].strip().split()[0]
                    mae = float(val_str)
                except Exception:
                    pass

            elif "Errore Relativo Medio (MRE)" in ln:
                # es: "Errore Relativo Medio (MRE) sull'Altitudine: 46.10 %"
                try:
                    val_str = ln.split(":")[-1].strip().split()[0]
                    mre = float(val_str)
                except Exception:
                    pass

    if mae is None or mre is None:
        print(f"[WARN] Missing MAE/MRE in {log_path}")
        return None

    return dict(
        mae=mae,
        mre=mre
    )


# ============================================================
# SCAN RUNS (fixed + adaptive)
# ============================================================

def analyze_runs(fixed_root, adapt_root, ignore_scenes=None):
    """
    Ritorna:
      per_scene_fixed:  scene -> [record_fixed]
      per_scene_adapt:  scene -> [record_adaptive_con_delta]
      baseline_by_scene: scene -> best_fixed_record (min MAE)
      config_list_adapt: sorted list of adaptive config_id (global)
    """
    if ignore_scenes is None:
        ignore_scenes = set()

    per_scene_fixed = defaultdict(list)
    per_scene_adapt = defaultdict(list)
    all_adapt_configs = set()

    # --------- FIXED ---------
    if fixed_root is not None:
        for folder in os.listdir(fixed_root):
            run_path = os.path.join(fixed_root, folder)
            if not os.path.isdir(run_path):
                continue

            parsed = parse_fixed_name(folder)
            if parsed is None:
                # print(f"[WARN] Cannot parse fixed run name: {folder}")
                continue

            scene = parsed["scene"]
            if scene in ignore_scenes:
                print(f"[INFO] Ignoring FIXED scene {scene}")
                continue


            log_path = os.path.join(run_path, "results.log")
            if not os.path.isfile(log_path):
                print(f"[WARN] results.log not found in fixed run: {run_path}")
                continue

            metrics = parse_results_log(log_path)
            if metrics is None:
                continue

            rec = dict(
                scene=parsed["scene"],
                type="fixed",
                config_id=parsed["config_id"],
                dt=parsed["dt"],
                C=parsed["C"],
                mae=metrics["mae"],
                mre=metrics["mre"],
            )
            per_scene_fixed[parsed["scene"]].append(rec)

    # --------- ADAPTIVE ---------
    if adapt_root is not None:
        for folder in os.listdir(adapt_root):
            run_path = os.path.join(adapt_root, folder)
            if not os.path.isdir(run_path):
                continue

            parsed = parse_adapt_name(folder)
            if parsed is None:
                # print(f"[WARN] Cannot parse adaptive run name: {folder}")
                continue

            scene = parsed["scene"]
            if scene in ignore_scenes:
                print(f"[INFO] Ignoring ADAPTIVE scene {scene}")
                continue

            log_path = os.path.join(run_path, "results.log")
            if not os.path.isfile(log_path):
                print(f"[WARN] results.log not found in adaptive run: {run_path}")
                continue

            metrics = parse_results_log(log_path)
            if metrics is None:
                continue

            rec = dict(
                scene=parsed["scene"],
                type="adaptive",
                config_id=parsed["config_id"],
                P=parsed["P"],
                I=parsed["I"],
                SP=parsed["SP"],
                TH=parsed["TH"],
                mae=metrics["mae"],
                mre=metrics["mre"],
            )
            per_scene_adapt[parsed["scene"]].append(rec)
            all_adapt_configs.add(rec["config_id"])

    # --------- Baseline fixed per scena (miglior MAE) ---------
    baseline_by_scene = {}
    for scene, runs in per_scene_fixed.items():
        if scene in ignore_scenes:
            continue
        if not runs:
            continue
        best = min(runs, key=lambda r: r["mae"])
        baseline_by_scene[scene] = best

    # --------- ΔMAE / ΔMRE per adaptive (wrt best fixed) ---------
    for scene, runs in per_scene_adapt.items():
        if scene in ignore_scenes:
            continue
        if scene not in baseline_by_scene:
            print(f"[WARN] No fixed baseline for scene {scene}, cannot compute deltas.")
            continue

        base = baseline_by_scene[scene]
        base_mae = base["mae"]
        base_mre = base["mre"]

        for r in runs:
            r["dMAE"] = r["mae"] - base_mae
            r["dMRE"] = r["mre"] - base_mre

    config_list_adapt = sorted(all_adapt_configs)
    return per_scene_fixed, per_scene_adapt, baseline_by_scene, config_list_adapt


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_scene_summary(per_scene_fixed, per_scene_adapt, baseline_by_scene, ignore_scenes=None):
    print("\n===================== PER-SCENE SUMMARY =====================")

    all_scenes = sorted(
        s for s in set(list(per_scene_fixed.keys()) + list(per_scene_adapt.keys()))
        if s not in ignore_scenes
    )

    for scene in all_scenes:
        print(f"\n[SCENE] {scene}")
        baseline = baseline_by_scene.get(scene, None)
        if baseline:
            print(f"  Baseline (fixed) best: {baseline['config_id']} | "
                  f"MAE={baseline['mae']:.4f} m | MRE={baseline['mre']:.2f} %")
        else:
            print("  Baseline (fixed): NONE")

        if scene in per_scene_fixed:
            print("  Fixed configs:")
            for r in sorted(per_scene_fixed[scene], key=lambda x: x["config_id"]):
                print(f"    {r['config_id']:>15} | MAE={r['mae']:.4f} m | MRE={r['mre']:.2f} %")

        if scene in per_scene_adapt:
            print("  Adaptive configs:")
            for r in sorted(per_scene_adapt[scene], key=lambda x: x["config_id"]):
                dMAE = r.get("dMAE", np.nan)
                dMRE = r.get("dMRE", np.nan)
                print(f"    {r['config_id']:>15} | MAE={r['mae']:.4f} m | MRE={r['mre']:.2f} %"
                      f" | dMAE={dMAE:+.4f} | dMRE={dMRE:+.2f} %")


# ============================================================
# SELEZIONE BEST CONFIGS PER SCENA
# ============================================================

def select_best_configs(per_scene_adapt, ignore_scenes=None):
    """
    Per ogni scena, seleziona:
      - best_dMAE: config con ΔMAE più negativo
      - best_dMRE: config con ΔMRE più negativo
      - best_tradeoff: distanza minima da (0,0) nel piano ΔMAE–ΔMRE
    """
    best_summary = {}

    for scene, runs in per_scene_adapt.items():
        if ignore_scenes and scene in ignore_scenes:
            continue
    
        valid = [r for r in runs if "dMAE" in r and "dMRE" in r]
        if not valid:
            continue

        # Best ΔMAE (più negativo)
        best_dMAE = min(valid, key=lambda x: x["dMAE"])

        # Best ΔMRE (più negativo)
        best_dMRE = min(valid, key=lambda x: x["dMRE"])

        # Best tradeoff => distanza minima dal punto (0,0)
        best_tradeoff = min(valid, key=lambda x: np.hypot(x["dMAE"], x["dMRE"]))

        best_summary[scene] = dict(
            best_dMAE=best_dMAE,
            best_dMRE=best_dMRE,
            best_tradeoff=best_tradeoff
        )

    return best_summary


def select_best_global_fixed(per_scene_fixed, ignore_scenes=None):
    """
    Trova la configurazione fixed_dt (dtX_cY) che:
        - compare nel maggior numero di scene
        - ha il MAE medio più basso
        - con minore varianza tra scene
    Restituisce:
        best_config_id, stats_dict
    """

    if ignore_scenes is None:
        ignore_scenes = set()

    # Aggrego per config_id
    agg = defaultdict(list)

    for scene, runs in per_scene_fixed.items():
        if scene in ignore_scenes:
            continue

        for r in runs:
            agg[r["config_id"]].append(r["mae"])

    best_cfg = None
    best_score = float("inf")
    stats_out = {}

    for cfg, maes in agg.items():
        if len(maes) < 2:
            continue  # troppo poco supporto per giudicarlo globale

        mean_mae = np.mean(maes)
        std_mae = np.std(maes)

        # criterio di selezione: media + penalità stabilità
        score = mean_mae + 0.2 * std_mae

        stats_out[cfg] = (mean_mae, std_mae, score)

        if score < best_score:
            best_score = score
            best_cfg = cfg

    return best_cfg, stats_out


def rank_adaptive_configs(per_scene_adapt, ignore_scenes=None):
    """
    Ranking globale delle configurazioni adaptive:
       - rewards improvements (negative dMAE)
       - penalizza degradazioni (positive dMAE)
       - ranked per consistenza across scenes
    Returns:
        sorted list of dicts: {config_id, score, n_improve, avg_gain, avg_loss}
    """

    if ignore_scenes is None:
        ignore_scenes = set()

    # ---- gather values grouped by config ----
    cfg_to_deltas = defaultdict(list)

    for scene, runs in per_scene_adapt.items():
        if scene in ignore_scenes:
            continue
        
        for r in runs:
            if "dMAE" in r:
                cfg_to_deltas[r["config_id"]].append(r["dMAE"])

    rankings = []

    for cfg, dvals in cfg_to_deltas.items():
        dvals = np.array(dvals)

        n_improve = np.sum(dvals < 0)
        n_worse   = np.sum(dvals > 0)

        avg_gain  = -np.mean(dvals[dvals < 0]) if np.any(dvals < 0) else 0.0
        avg_loss  =  np.mean(dvals[dvals > 0]) if np.any(dvals > 0) else 0.0

        # score = improvements - penalty
        score = (n_improve * avg_gain) - (n_worse * avg_loss)

        rankings.append(dict(
            config_id = cfg,
            score     = score,
            n_improve = int(n_improve),
            n_worse   = int(n_worse),
            avg_gain  = avg_gain,
            avg_loss  = avg_loss
        ))

    rankings.sort(key=lambda x: x["score"], reverse=True)
    return rankings


def print_adaptive_ranking(rankings, top_k=None):
    """
    Print ranking results in readable form.
    """
    print("\n===================== GLOBAL ADAPTIVE CONFIG RANKING =====================")
    print(f"{'Config':>20} | {'Score':>8} | {'Improves':>9} | {'Worse':>6} | {'AvgGain':>8} | {'AvgLoss':>8}")
    print("-" * 78)

    n = len(rankings) if top_k is None else min(top_k, len(rankings))

    for r in rankings[:n]:
        print(f"{r['config_id']:>20} | "
              f"{r['score']:8.3f} | "
              f"{r['n_improve']:9d} | "
              f"{r['n_worse']:6d} | "
              f"{r['avg_gain']:8.3f} | "
              f"{r['avg_loss']:8.3f}")



# ============================================================
# PLOTTING
# ============================================================
def plot_global_multiscene(per_scene_fixed, per_scene_adapt, baseline_by_scene, out_path, ignore_scenes=None):
    """
    Create ONE global figure:
      - 1 row per scene
      - shows fixed vs adaptive MAE + MRE
      - includes per-row legends so plots are self-explanatory
    """

    scenes = sorted(set(list(per_scene_fixed.keys()) + list(per_scene_adapt.keys())))
    if ignore_scenes:
        scenes = [s for s in scenes if s not in ignore_scenes]

    if not scenes:
        print("[WARN] No scenes available for global plot.")
        return

    n = len(scenes)

    fig, axes = plt.subplots(n, 1, figsize=(26, 3.5*n), sharex=False)

    if n == 1:
        axes = [axes]

    for i, scene in enumerate(scenes):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        fixed_runs = sorted(per_scene_fixed.get(scene, []), key=lambda x: x["config_id"])
        adapt_runs = sorted(per_scene_adapt.get(scene, []), key=lambda x: x["config_id"])

        cfg_fixed = [r["config_id"] for r in fixed_runs]
        cfg_adapt = [r["config_id"] for r in adapt_runs]

        mae_fixed = [r["mae"] for r in fixed_runs]
        mae_adapt = [r["mae"] for r in adapt_runs]

        mre_fixed = [r["mre"] for r in fixed_runs]
        mre_adapt = [r["mre"] for r in adapt_runs]

        x_fixed = np.arange(len(cfg_fixed))
        x_adapt = np.arange(len(cfg_adapt)) + len(cfg_fixed)

        lines = []

        # === MAE LINES ===
        if cfg_fixed:
            l1, = ax1.plot(x_fixed, mae_fixed, "o-", color="tab:blue", label="MAE Fixed")
            lines.append(l1)

        if cfg_adapt:
            l2, = ax1.plot(x_adapt, mae_adapt, "o-", color="tab:orange", label="MAE Adaptive")
            lines.append(l2)

        # Baseline MAE
        if scene in baseline_by_scene:
            b = baseline_by_scene[scene]
            l3 = ax1.axhline(b["mae"], linestyle="--", color="tab:blue", alpha=0.5, label="MAE Baseline")
            lines.append(l3)

        # === MRE LINES ===
        if cfg_fixed:
            l4, = ax2.plot(x_fixed, mre_fixed, "s--", color="tab:green", label="MRE Fixed")
            lines.append(l4)

        if cfg_adapt:
            l5, = ax2.plot(x_adapt, mre_adapt, "s--", color="tab:red", label="MRE Adaptive")
            lines.append(l5)

        # baseline MRE
        if scene in baseline_by_scene:
            b = baseline_by_scene[scene]
            l6 = ax2.axhline(b["mre"], linestyle="--", color="tab:green", alpha=0.5, label="MRE Baseline")
            lines.append(l6)

        ax1.set_title(scene, fontsize=12)
        ax1.set_ylabel("MAE (m)", color="tab:blue")
        ax2.set_ylabel("MRE (%)", color="tab:red")
        ax1.grid(True, linestyle=":")

        xticks = np.concatenate([x_fixed, x_adapt])
        xlabels = cfg_fixed + cfg_adapt
        if len(xticks) > 0:
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xlabels, rotation=90)

        # === Add legend per row ===
        # Only show unique labels
        handles, labels = [], []
        for h in ax1.get_lines() + ax2.get_lines():
            if h.get_label() not in labels:
                handles.append(h)
                labels.append(h.get_label())

        ax1.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    global_fig = os.path.join(out_path, "global_multiscene_fixed_vs_adaptive.png")
    fig.savefig(global_fig, dpi=300)
    print(f"[PLOT] Saved → {global_fig}")



def plot_absolute_per_scene(per_scene_fixed, per_scene_adapt, out_dir, ignore_scenes=None):
    """
    Per ogni scena:
      - subplot 1: MAE (m) per tutte le config (fixed + adaptive)
      - subplot 2: MRE (%) per tutte le config (fixed + adaptive)
    """
    os.makedirs(out_dir, exist_ok=True)

    all_scenes = sorted(set(list(per_scene_fixed.keys()) + list(per_scene_adapt.keys())))

    for scene in all_scenes:
        if ignore_scenes and scene in ignore_scenes:
            continue

        fixed_runs = sorted(per_scene_fixed.get(scene, []), key=lambda x: x["config_id"])
        adapt_runs = sorted(per_scene_adapt.get(scene, []), key=lambda x: x["config_id"])

        labels_fixed = [r["config_id"] for r in fixed_runs]
        labels_adapt = [r["config_id"] for r in adapt_runs]

        mae_fixed = [r["mae"] for r in fixed_runs]
        mre_fixed = [r["mre"] for r in fixed_runs]
        mae_adapt = [r["mae"] for r in adapt_runs]
        mre_adapt = [r["mre"] for r in adapt_runs]

        nF = len(labels_fixed)
        nA = len(labels_adapt)

        # Asse x: prima i fixed, poi gli adaptive
        x_fixed = np.arange(nF)
        x_adapt = np.arange(nA) + nF

        fig, (ax_mae, ax_mre) = plt.subplots(2, 1, figsize=(22, 10), sharex=True)

        # --- MAE ---
        if nF > 0:
            ax_mae.plot(x_fixed, mae_fixed, "o-", label="Fixed dt", color="tab:blue")
        if nA > 0:
            ax_mae.plot(x_adapt, mae_adapt, "s-", label="Adaptive", color="tab:orange")

        ax_mae.set_ylabel("MAE Altitude (m)")
        ax_mae.set_title(f"{scene} – Absolute Altitude Error (MAE/MRE) – Fixed vs Adaptive")
        ax_mae.grid(True, linestyle=":")
        ax_mae.legend()

        # --- MRE ---
        if nF > 0:
            ax_mre.plot(x_fixed, mre_fixed, "o-", label="Fixed dt", color="tab:blue")
        if nA > 0:
            ax_mre.plot(x_adapt, mre_adapt, "s-", label="Adaptive", color="tab:orange")

        ax_mre.set_ylabel("MRE Altitude (%)")
        ax_mre.set_xlabel("Configuration index (Fixed + Adaptive)")
        ax_mre.grid(True, linestyle=":")
        ax_mre.legend()

        # Tick labels: prima fixed, poi adaptive
        xticks = np.concatenate([x_fixed, x_adapt])
        xlabels = labels_fixed + labels_adapt
        ax_mre.set_xticks(xticks)
        ax_mre.set_xticklabels(xlabels, rotation=90)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{scene}_absolute_fixed_vs_adaptive.png")
        fig.savefig(out_path, dpi=250)
        plt.close(fig)
        print(f"[PLOT] Saved absolute MAE/MRE plot → {out_path}")


def plot_deltas_per_scene(per_scene_adapt, out_dir, ignore_scenes=None):
    """
    Per ogni scena:
      - plot a 2 assi (ΔMAE in blu, ΔMRE in rosso) vs config_id (asse x).
    """
    os.makedirs(out_dir, exist_ok=True)

    for scene, runs in per_scene_adapt.items():
        if ignore_scenes and scene in ignore_scenes:
            continue

        runs_sorted = sorted(runs, key=lambda x: x["config_id"])
        if not runs_sorted:
            continue

        cfgs = [r["config_id"] for r in runs_sorted]
        dMAE = [r.get("dMAE", np.nan) for r in runs_sorted]
        dMRE = [r.get("dMRE", np.nan) for r in runs_sorted]

        x = np.arange(len(cfgs))

        fig, ax1 = plt.subplots(figsize=(22, 6))
        ax2 = ax1.twinx()

        ax1.plot(x, dMAE, "o-", color="tab:blue", label="ΔMAE")
        ax2.plot(x, dMRE, "s-", color="tab:red", label="ΔMRE")

        ax1.axhline(0.0, linestyle="--", color="tab:blue", linewidth=1)
        ax2.axhline(0.0, linestyle="--", color="tab:red", linewidth=1)

        ax1.set_ylabel("ΔMAE (m) – negative = adaptive better", color="tab:blue")
        ax2.set_ylabel("ΔMRE (%) – negative = adaptive better", color="tab:red")
        ax1.set_title(f"{scene} – ΔMAE / ΔMRE vs Best Fixed Baseline")
        ax1.grid(True, linestyle=":")

        ax1.set_xticks(x)
        ax1.set_xticklabels(cfgs, rotation=90)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{scene}_delta_lines.png")
        fig.savefig(out_path, dpi=250)
        plt.close(fig)
        print(f"[PLOT] Saved delta line plot → {out_path}")


def plot_delta_scatter(per_scene_adapt, out_dir, ignore_scenes=None):
    """
    Per ogni scena:
      - scatter ΔMAE vs ΔMRE, annotando le config (Pareto view).
    """
    os.makedirs(out_dir, exist_ok=True)

    for scene, runs in per_scene_adapt.items():
        if ignore_scenes and scene in ignore_scenes:
            continue

        valid = [r for r in runs if "dMAE" in r and "dMRE" in r]
        if not valid:
            continue

        dMAE = [r["dMAE"] for r in valid]
        dMRE = [r["dMRE"] for r in valid]
        labels = [r["config_id"] for r in valid]

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(dMAE, dMRE, c="tab:purple", alpha=0.7)

        plt.axvline(0.0, linestyle="--", color="black", linewidth=1)
        plt.axhline(0.0, linestyle="--", color="black", linewidth=1)

        for x, y, lb in zip(dMAE, dMRE, labels):
            plt.annotate(lb, (x, y), fontsize=6)

        plt.title(f"{scene} – ΔMAE vs ΔMRE (Adaptive – Fixed)")
        plt.xlabel("ΔMAE (m)  [negative = better]")
        plt.ylabel("ΔMRE (%) [negative = better]")
        plt.grid(True, linestyle=":")

        out_path = os.path.join(out_dir, f"{scene}_delta_scatter.png")
        plt.savefig(out_path, dpi=250)
        plt.close(fig)
        print(f"[PLOT] Saved delta scatter plot → {out_path}")


def plot_best_configs(per_scene_adapt, best_summary, out_dir, ignore_scenes=None):
    """
    Per ogni scena:
      - scatter ΔMAE vs ΔMRE
      - evidenzia:
          * Best ΔMAE      (stella rossa)
          * Best ΔMRE      (stella verde)
          * Best tradeoff  (rombo blu)
    """
    os.makedirs(out_dir, exist_ok=True)

    for scene, runs in per_scene_adapt.items():
        if ignore_scenes and scene in ignore_scenes:
            continue
        if scene not in best_summary:
            continue

        valid = [r for r in runs if "dMAE" in r and "dMRE" in r]
        if not valid:
            continue

        dMAE = [r["dMAE"] for r in valid]
        dMRE = [r["dMRE"] for r in valid]
        labels = [r["config_id"] for r in valid]

        b = best_summary[scene]
        bA = b["best_dMAE"]
        bR = b["best_dMRE"]
        bT = b["best_tradeoff"]

        fig = plt.figure(figsize=(8, 8))

        plt.scatter(dMAE, dMRE, s=50, alpha=0.5, label="All adaptive configs")

        plt.scatter([bA["dMAE"]], [bA["dMRE"]],
                    s=200, marker="*", color="red", label="Best ΔMAE")
        plt.scatter([bR["dMAE"]], [bR["dMRE"]],
                    s=200, marker="*", color="green", label="Best ΔMRE")
        plt.scatter([bT["dMAE"]], [bT["dMRE"]],
                    s=160, marker="D", color="blue", label="Best tradeoff")

        plt.axvline(0.0, linestyle="--", color="black", linewidth=1)
        plt.axhline(0.0, linestyle="--", color="black", linewidth=1)

        for x, y, lb in zip(dMAE, dMRE, labels):
            plt.annotate(lb, (x, y), fontsize=6, alpha=0.8)

        plt.title(f"{scene} – Best Adaptive Configs (ΔMAE vs ΔMRE)")
        plt.xlabel("ΔMAE (m)  [negative = better]")
        plt.ylabel("ΔMRE (%) [negative = better]")
        plt.grid(True, linestyle=":")
        plt.legend(loc="best")

        out_path = os.path.join(out_dir, f"{scene}_best_configs.png")
        plt.savefig(out_path, dpi=250)
        plt.close(fig)
        print(f"[PLOT] Saved best-config scatter → {out_path}")


def plot_selected_baseline(per_scene_fixed, per_scene_adapt, selected_fixed_cfg, out_dir, ignore_scenes=None):
    """
    Plot per scena:
        - baseline scelta (dt selezionato globalmente)
        - rispetto a tutte le adaptive configs
    """

    os.makedirs(out_dir, exist_ok=True)

    for scene in sorted(set(list(per_scene_fixed.keys()) + list(per_scene_adapt.keys()))):

        if ignore_scenes and scene in ignore_scenes:
            continue

        fixed_runs = per_scene_fixed.get(scene, [])
        adapt_runs = per_scene_adapt.get(scene, [])

        # trova valore baseline per questa scena
        base_val = None
        for r in fixed_runs:
            if r["config_id"] == selected_fixed_cfg:
                base_val = r["mae"]
                break

        if base_val is None:
            # questa scena non ha quella baseline, skip
            continue

        labels_adapt = [r["config_id"] for r in adapt_runs]
        mae_adapt = [r["mae"] for r in adapt_runs]

        fig, ax = plt.subplots(figsize=(18, 5))

        # baseline come linea orizzontale
        ax.axhline(base_val, color="tab:blue", linestyle="--", linewidth=2,
                   label=f"fixed baseline = {selected_fixed_cfg}")

        # adaptive points
        if adapt_runs:
            ax.plot(labels_adapt, mae_adapt, "o-", color="tab:orange",
                    label="adaptive configs")

        ax.set_title(f"{scene} — Baseline vs Adaptive")
        ax.set_ylabel("MAE Altitude (m)")
        ax.set_xticklabels(labels_adapt, rotation=90)
        ax.grid(True, linestyle=":")
        ax.legend()

        out_path = os.path.join(out_dir, f"{scene}_selected_baseline_vs_adaptive.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=250)
        plt.close(fig)

        print(f"[PLOT] Saved baseline comparison → {out_path}")



# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_dir", type=str, default="results_fpv/runs_fpv_fixed/",
                        help="Directory with FIXED dt runs (fpv_*_dtXX_cY.Y)")
    parser.add_argument("--adapt_dir", type=str, default="results_fpv/runs_fpv_adaptive/",
                        help="Directory with ADAPTIVE slicing runs (fpv_*_P..._sp..._th...)")
    parser.add_argument("--out_dir", type=str, default="analysis_fpv_fixed_vs_adaptive/",
                        help="Output directory for plots")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Optional list of scene names to ignore, e.g. --ignore indoor_45_9 outdoor_45_1"
    )

    args = parser.parse_args()

    per_scene_fixed, per_scene_adapt, baseline_by_scene, config_list_adapt = analyze_runs(
        fixed_root=args.fixed_dir,
        adapt_root=args.adapt_dir,
        ignore_scenes=set(args.ignore)
    )

    # Stampa riassunto
    print_scene_summary(per_scene_fixed, per_scene_adapt, baseline_by_scene, ignore_scenes=set(args.ignore))

    # Seleziona best config per scena (sui delta)
    best_summary = select_best_configs(per_scene_adapt, ignore_scenes=set(args.ignore))
    
    # === NEW: select globally best fixed configuration ===
    best_fixed_cfg, stats = select_best_global_fixed(per_scene_fixed, ignore_scenes=set(args.ignore))


    rankings = rank_adaptive_configs(per_scene_adapt, ignore_scenes=set(args.ignore))
    print_adaptive_ranking(rankings, top_k=None)

    print("\n===================== GLOBAL BEST FIXED DT CONFIG =====================")
    print(f"Selected fixed baseline config = {best_fixed_cfg}")
    for cfg, (mean_mae, std_mae, score) in stats.items():
        print(f"  {cfg}: mean={mean_mae:.3f}, std={std_mae:.3f}, score={score:.3f}")

        
    
    print("\n===================== BEST ADAPTIVE CONFIGS PER SCENE =====================")
    for scene in sorted(best_summary.keys()):
        b = best_summary[scene]
        print(f"\n[SCENE] {scene}")
        print(f"  -> Best ΔMAE     : {b['best_dMAE']['config_id']}  "
              f"dMAE={b['best_dMAE']['dMAE']:+.4f}, dMRE={b['best_dMAE']['dMRE']:+.2f}%")
        print(f"  -> Best ΔMRE     : {b['best_dMRE']['config_id']}  "
              f"dMAE={b['best_dMRE']['dMAE']:+.4f}, dMRE={b['best_dMRE']['dMRE']:+.2f}%")
        print(f"  -> Best tradeoff : {b['best_tradeoff']['config_id']}  "
              f"dMAE={b['best_tradeoff']['dMAE']:+.4f}, dMRE={b['best_tradeoff']['dMRE']:+.2f}%")

    # Plot:
    plot_absolute_per_scene(per_scene_fixed, per_scene_adapt, args.out_dir, ignore_scenes=set(args.ignore))
    plot_deltas_per_scene(per_scene_adapt, args.out_dir, ignore_scenes=set(args.ignore))
    plot_delta_scatter(per_scene_adapt, args.out_dir, ignore_scenes=set(args.ignore))
    plot_best_configs(per_scene_adapt, best_summary, args.out_dir, ignore_scenes=set(args.ignore))
    # === NEW GLOBAL MULTISCENE FIGURE ===
    plot_global_multiscene(
        per_scene_fixed,
        per_scene_adapt,
        baseline_by_scene,
        args.out_dir,
        ignore_scenes=set(args.ignore)
    )

        # === NEW: plot selected baseline vs all adaptives ===
    plot_selected_baseline(
        per_scene_fixed,
        per_scene_adapt,
        selected_fixed_cfg=best_fixed_cfg,
        out_dir=args.out_dir,
        ignore_scenes=set(args.ignore)
    )
if __name__ == "__main__":
    main()
