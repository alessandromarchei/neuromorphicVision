import torch
import numpy as np
import glob
import cv2
import os.path as osp
import torch.utils.data as data
import h5py
import os
import torch.nn.functional as F
from tqdm import tqdm
import bisect
import matplotlib.pyplot as plt

from testing.utils.event_utils import to_event_frame


# Default limits (can be overridden in function params)
DEFAULT_MAX_EVENTS_LOADED = 1_000_000
DEFAULT_BATCH_EVENTS = 200_000


#valid ranges for each MVSEC scene, skipping frames where the drone is not moving (before landing and takeoff)

VALID_FRAME_RANGES = {
    "indoor_flying1": (180, 2130),  #TODO: change after the debugging adaptive slicing
    "indoor_flying2": (250, 2560),
    "indoor_flying3": (200, 2850),
    "indoor_flying4": (150, 580),
    "outdoor_day1":   (0, 11750)
}



def get_valid_range_from_scene(scenedir):
    scenedir_lower = scenedir.lower()
    for key, rng in VALID_FRAME_RANGES.items():
        if key in scenedir_lower:
            print(f"[MVSEC] Auto-valid range detected for {key}: {rng}")
            return rng
    return None


def dvxplorer_evs_iterator(
    scenedir,
    dT_ms=10.0,
    H=480,
    W=640,
    max_events_loaded=DEFAULT_MAX_EVENTS_LOADED,
    batch_events=DEFAULT_BATCH_EVENTS,
    use_valid_frame_range=False,    
):
    """
    Streaming iterator that yields tuples:

        event_frame,
        t_event_us,
        dt_ms,

    """

    print(f"[MVSEC] Streaming from {scenedir}, dT_ms={dT_ms}")

    # Auto-detect valid frame range from scene name
    if use_valid_frame_range is True:
        valid_frame_range = get_valid_range_from_scene(scenedir)
    else:
        valid_frame_range = None
    # ---------------------------------------------------------
    # Load EVENTS from *.hdf5
    # ---------------------------------------------------------
    h5_main = glob.glob(os.path.join(scenedir, "*.hdf5"))
    assert len(h5_main) == 1
    f_ev = h5py.File(h5_main[0], "r")
    evs = f_ev[f"dvxplorermicro/events"]
    N = evs["x"].shape[0]

    print("[HDF5] Top-level groups:", list(f_ev.keys()))
    for g in f_ev.keys():
        print(f"[HDF5] Group '{g}' contains:", list(f_ev[g].keys()))

    evs = f_ev["dvxplorermicro/events"]
    print("[HDF5] Events datasets:", list(evs.keys()))

    # ---------------------------------------------------------
    # EVENT STREAMING BUFFER
    # ---------------------------------------------------------
    x_buf = np.empty(0, dtype=np.uint16)
    y_buf = np.empty(0, dtype=np.uint16)
    ts_us_buf = np.empty(0, dtype=np.int64)
    pol_buf = np.empty(0, dtype=np.int8)

    abs_start = 0
    load_cursor = 0

    def load_more():
        """Load next event batch from HDF5 (x,y,t,p are separate datasets)."""
        nonlocal load_cursor, x_buf, y_buf, ts_us_buf, pol_buf

        if load_cursor >= N:
            return False

        end = min(load_cursor + batch_events, N)

        # HDF5 datasets
        x = evs["x"][load_cursor:end].astype(np.uint16)
        y = evs["y"][load_cursor:end].astype(np.uint16)
        ts = evs["t"][load_cursor:end].astype(np.int64)   # microseconds already? If seconds, convert.
        p  = evs["p"][load_cursor:end].astype(np.int8)

        # Append
        x_buf = np.concatenate([x_buf, x])
        y_buf = np.concatenate([y_buf, y])
        ts_us_buf = np.concatenate([ts_us_buf, ts])       # or convert if necessary
        pol_buf = np.concatenate([pol_buf, p])

        load_cursor = end
        return True

    def trim_buffer():
        nonlocal abs_start, x_buf, y_buf, ts_us_buf, pol_buf
        extra = len(x_buf) - max_events_loaded
        if extra > 0:
            x_buf = x_buf[extra:]
            y_buf = y_buf[extra:]
            ts_us_buf = ts_us_buf[extra:]
            pol_buf = pol_buf[extra:]
            abs_start += extra

    # =======================================================================
    # FIXED TEMPORAL SLICING (dT_ms > 0)
    # =======================================================================
    print(f"[DVXplorer] Fixed slicing: every {dT_ms} ms")

    if load_cursor == 0:
        load_more()

    dt_us = int(dT_ms * 1000)
    t0_us = ts_us_buf[0]
    t_end_us = int(float(evs["t"][-1]))

    frame_idx = 0
    while t0_us < t_end_us:

        if valid_frame_range is not None:
            start_i, end_i = valid_frame_range
            if frame_idx < start_i:
                frame_idx += 1
                t0_us += dt_us
                continue
            if frame_idx > end_i:
                break


        t1_us = t0_us + dt_us

        while len(ts_us_buf) == 0 or ts_us_buf[-1] < t1_us:
            if not load_more():
                break
            trim_buffer()

        start = np.searchsorted(ts_us_buf, t0_us, side="left")
        end   = np.searchsorted(ts_us_buf, t1_us, side="left")

        if end > start:
            bx = x_buf[start:end]
            by = y_buf[start:end]
            bp = pol_buf[start:end]


            event_frame = to_event_frame(bx, by, bp, H, W)

            dt_ms_here = (ts_us_buf[end - 1] - ts_us_buf[start]) * 1e-3

            yield event_frame, t0_us, dt_ms_here

        t0_us = t1_us
        trim_buffer()

        frame_idx += 1

    f_ev.close()
