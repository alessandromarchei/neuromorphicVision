from src.scripts.VisionNodeEventsPlayback import VisionNodeEventsPlayback
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,
                        help="Experiment name. Will create runs/<name>/ for logs and plots.")
    parser.add_argument("--yaml", type=str, default="config/config_mvsec.yaml",
                        help="YAML config path.")

    args = parser.parse_args()

    # Create run folder
    run_dir = os.path.join("runs", args.run)
    os.makedirs(run_dir, exist_ok=True)

    # Copy YAML inside run folder for reproducibility
    os.system(f"cp {args.yaml} {os.path.join(run_dir, 'config_used.yaml')}")

    # Create logger file
    log_path = os.path.join(run_dir, "logs.log")
    log_file = open(log_path, "w")

    # Initialize node
    node = VisionNodeEventsPlayback(args.yaml)

    # Redirect evaluation prints to both console and log file
    def dual_print(s):
        print(s)
        log_file.write(s + "\n")
        log_file.flush()

    node.run()

    # Clean NaNs
    node.cleanNaNEntries()

    # Final report (also printed to log)
    final_report = []
    def capture_print(*msgs):
        txt = " ".join(str(m) for m in msgs)
        final_report.append(txt)

    # temporarily hijack print inside printFinalReport
    import builtins
    old_print = builtins.print
    builtins.print = capture_print
    node.printFinalReport()
    builtins.print = old_print

    for line in final_report:
        dual_print(line)

    # ---- PLOT AEE AND OUTLIERS ----
    frames = np.arange(len(node.eval_AEE))

    plt.figure(figsize=(13,6))
    plt.plot(frames, node.eval_AEE, label="AEE", linewidth=2)
    plt.plot(frames, np.array(node.eval_outliers)*100.0, label="Outliers (%)", linewidth=2)

    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title(f"AEE and Outlier Rate Over Time – {args.run}")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(run_dir, "plot_aee.png")
    plt.savefig(plot_path)
    plt.close()

 # ---- SAVE CSV ----
    import csv

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "frame_id",
            "timestamp_us",
            "AEE",
            "outlier_percentage",
            "dt_ms",
            "dt_gt_ms",
            "N_points"
        ])

        for i in range(len(node.eval_AEE)):
            writer.writerow([
                node.eval_frameID[i],
                node.eval_timestamp[i],
                node.eval_AEE[i],
                node.eval_outliers[i],
                node.eval_dt_ms[i],
                node.eval_dtgt_ms[i],
                node.eval_Npoints[i],
            ])

    dual_print(f"Saved CSV to: {csv_path}")


    log_file.close()

    print("All done.")
