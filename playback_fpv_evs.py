from src.scripts.VisionNodeUZHFPVEventsPlayback import VisionNodeUZHFPVEventsPlayback
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=False,
                        help="Experiment name used for saving logs and plots inside runs/<name>/")
    parser.add_argument("--yaml", type=str, default="config/config_fpv.yaml",
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
    node = VisionNodeUZHFPVEventsPlayback(args.yaml)
    node.run()

    # capture final report printing
    final_report = []

    def capture_print(*msgs):
        txt = " ".join(str(m) for m in msgs)
        final_report.append(txt)

    # print final report normally
    for line in final_report:
        dual_print(line)

        
