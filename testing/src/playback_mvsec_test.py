#!/usr/bin/env python3

import os
import cv2
import csv
import math
import time
import yaml  # PyYAML
import numpy as np
import threading

from testing.utils.load_utils_mvsec import mvsec_evs_iterator
from testing.utils.eval_utils import run_event_frame
# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    mvsec = "/home/alessandro/datasets/mvsec/indoor_flying/indoor_flying4_data/"
    
    run_event_frame(mvsec, viz=True, iterator=mvsec_evs_iterator(mvsec, side="left", dT_ms=100))
    print("All done.")
