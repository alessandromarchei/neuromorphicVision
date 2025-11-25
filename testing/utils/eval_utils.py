
import os
import torch
from pathlib import Path
import yaml
import copy
import shutil
from scipy.spatial.transform import Rotation as R
from tabulate import tabulate

from tqdm import tqdm

from testing.utils.viz_utils import visualize_event_frame

@torch.no_grad()
def run_event_frame(events_dir, viz=False, iterator=None, H=260, W=346): 
    
    for i, (event_frame, t) in enumerate(tqdm(iterator)):
        #each iterator is a tuple (event_frame, timestamp)
        print(f"Processing event frame {i} at time {t} us with shape {event_frame.shape}")

        visualize_event_frame(event_frame, True, 0)
        
    return event_frame
