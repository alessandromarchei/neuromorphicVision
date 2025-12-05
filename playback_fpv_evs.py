from src.scripts.VisionNodeUZHFPVEventsPlayback import VisionNodeUZHFPVEventsPlayback
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=False, default="fpv_evs_playback",
                        help="Experiment name used for saving logs and plots inside runs/<name>/")
    parser.add_argument("--yaml", type=str, default="config/config_fpv.yaml",
                        help="YAML config path.")
    parser.add_argument("--out_dir", type=str, default="results_fpv",
                    help="Output directory for saving logs and plots.")
    args = parser.parse_args()

    # -------------------------------------------------------
    # 2) RUN VALIDATION PIPELINE
    # -------------------------------------------------------
    node = VisionNodeUZHFPVEventsPlayback(args.yaml, run_id=args.run, out_dir=args.out_dir)
    node.run()

    node.log()