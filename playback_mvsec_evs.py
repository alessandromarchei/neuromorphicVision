from src.scripts.VisionNodeEventsPlayback import VisionNodeEventsPlayback
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=False,
                        help="Experiment name used for saving logs and plots inside runs/<name>/")
    parser.add_argument("--yaml", type=str, default="config/config_mvsec.yaml",
                        help="YAML config path.")
    parser.add_argument("--out_dir", type=str, default="runs",
                    help="Output directory for saving logs and plots.")
    args = parser.parse_args()

    # -------------------------------------------------------
    # 1) RUN DIRECTORY HANDLING  (OPTIONAL)
    # -------------------------------------------------------
    if args.run:
        run_dir = os.path.join(args.out_dir, args.run)
        os.makedirs(run_dir, exist_ok=True)

        # save YAML for reproducibility
        os.system(f"cp {args.yaml} {os.path.join(run_dir, 'config.yaml')}")

        # create log file
        log_path = os.path.join(run_dir, "logs.log")
        log_file = open(log_path, "w")

        def dual_print(s):
            print(s)
            log_file.write(s + "\n")
            log_file.flush()

    else:
        run_dir = None
        log_file = None

        # no disk logging
        def dual_print(s):
            print(s)

    # -------------------------------------------------------
    # 2) RUN VALIDATION PIPELINE
    # -------------------------------------------------------
    node = VisionNodeEventsPlayback(args.yaml)
    node.run()
    node.cleanNaNEntries()

    # capture final report printing
    final_report = []

    def capture_print(*msgs):
        txt = " ".join(str(m) for m in msgs)
        final_report.append(txt)

    import builtins
    old_print = builtins.print
    builtins.print = capture_print
    node.printFinalReport()
    builtins.print = old_print

    # print final report normally
    for line in final_report:
        dual_print(line)

    # -------------------------------------------------------
    # 3) SAVE RESULTS ONLY IF --run PROVIDED
    # -------------------------------------------------------
    if run_dir is not None:
        # ========= PLOT =========
        import matplotlib.pyplot as plt
        import numpy as np

        frames = np.arange(len(node.eval_AEE))

        plt.figure(figsize=(13, 6))
        plt.plot(frames, node.eval_AEE, label="AEE", linewidth=2)
        plt.plot(frames, np.array(node.eval_outliers) * 100.0,
                 label="Outliers (%)", linewidth=2)
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.title(f"AEE and Outlier Rate Over Time – {args.run}")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(run_dir, "plot_aee.png")
        plt.savefig(plot_path)
        plt.close()
        dual_print(f"Saved plot to: {plot_path}")

        # ========= CSV =========
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
