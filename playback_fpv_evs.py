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
    parser.add_argument("--out_dir", type=str, default="results_fpv",
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

    else:
        run_dir = None
        log_file = None

    # -------------------------------------------------------
    # 2) RUN VALIDATION PIPELINE
    # -------------------------------------------------------
    node = VisionNodeUZHFPVEventsPlayback(args.yaml, run_id=args.run, out_dir=args.out_dir)
    node.run()

    node.log()