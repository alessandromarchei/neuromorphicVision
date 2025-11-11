#!/usr/bin/env python3

import os
import cv2
import csv
import math
import time
import yaml  # PyYAML
import numpy as np
import threading

from testing.utils.load_utils import fpv_evs_iterator
from testing.utils.eval_utils import run_voxel
# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    fpv_scene_dir = "/home/alessandro/Politecnico Di Torino Studenti Dropbox/Alessandro Marchei/paper/lis_nv/fpv/indoor_45_12_davis_with_gt"
    
    run_voxel(fpv_scene_dir, viz=True, iterator=fpv_evs_iterator(fpv_scene_dir))
    print("All done.")
