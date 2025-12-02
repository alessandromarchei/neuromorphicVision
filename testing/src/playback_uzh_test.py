#!/usr/bin/env python3

import os
import cv2
import csv
import math
import time
import yaml  # PyYAML
import numpy as np
import threading

from testing.utils.load_utils_fpv import fpv_evs_iterator
from testing.utils.eval_utils import run_event_frame
# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    fpv_scene_dir = "/home/alessandro/datasets/fpv/indoor_45_13_davis_with_gt"
    
    run_event_frame(fpv_scene_dir, viz=True, iterator=fpv_evs_iterator(fpv_scene_dir))
    print("All done.")
