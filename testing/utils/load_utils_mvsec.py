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

from testing.utils.event_utils import to_event_frame

"""
limit to prevent OOM error on CPU RAM. 
1 event is 4xdouble => 4x8 bytes => 32 bytes per event 
Scene           Time[s]     MER[events/second]  Nevents 
Indoor 1        70.4        185488              13058355 
indoor 2        84.5        273567              23116411
Indoor 3        94.7        243953              23102349
Indoor 4        20.0        361579              7231580
OutdoorDay 1    262         386178              101178636 

Summary of memory:
Original MVSEC = float64 for x,y,t,p → 32 bytes/event
Our representation:
  x,y  → uint16 (4 bytes total)
  t    → float64 (8 bytes)
  p    → int8   (1 byte)

Total ≈ 13 bytes/event instead of 32 bytes/event.
"""

MAX_EVENTS_LOADED = 10


def read_rmap(rect_file, H=180, W=240):
    h5file = glob.glob(rect_file)[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    assert rectify_map.shape == (H, W, 2)
    rmap.close()
    return rectify_map


# ======================================================================
#   MVSEC EVENT ITERATOR (NO-ALL_EVS VERSION)
# ======================================================================
def mvsec_evs_iterator(scenedir, side="left", dT_ms=None, H=260, W=346, rectify=True):

    print(f"[MVSEC] Loading MVSEC events from {scenedir}")

    # ============================
    # LOAD INPUTS
    # ============================
    h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
    assert len(h5in) == 1
    datain = h5py.File(h5in[0], 'r')

    evs = datain["davis"][side]["events"]  # HDF5 dataset

    # Load per-column (NO OOM)
    x  = evs[:, 0].astype(np.uint16)
    y  = evs[:, 1].astype(np.uint16)
    ts = evs[:, 2].astype(np.float64)
    pol = evs[:, 3].astype(np.int8)

    N = len(ts)

    # Image timestamps (used only if dT_ms=None)
    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"tss_imgs_us_{side}.txt")))
    num_imgs = len(tss_imgs_us)
    print(f"MVSEC sample points : {num_imgs}")

    # Rectification map
    rectify_map = read_rmap(osp.join(scenedir, f"rectify_map_{side}.h5"), H=H, W=W)

    # Debug dataset contents
    print(f"[MVSEC] Available datasets in HDF5 file: {list(datain['davis'][side].keys())}")

    # Mapping from images to events (only used if dT_ms=None)
    event_idxs = datain["davis"][side]["image_raw_event_inds"][:]

    datain.close()

    # ==================================================================
    # CASE 1 — DEFAULT MVSEC SLICING
    # ==================================================================
    if dT_ms is None:
        print("[MVSEC] Using default MVSEC slicing")

        evidx_left = 0

        for img_i in range(num_imgs):

            evid_next = event_idxs[img_i]

            # Build batch slices WITHOUT all_evs
            bx = x[evidx_left:evid_next]
            by = y[evidx_left:evid_next]
            bts = ts[evidx_left:evid_next]
            bp = pol[evidx_left:evid_next]

            evidx_left = evid_next

            if len(bts) == 0:
                continue

            # Rectify coordinates
            if not rectify:
                rect = np.stack([bx, by], axis=-1)
            else:
                rect = rectify_map[
                    by.astype(np.int32),
                    bx.astype(np.int32)
                ]

            # Build event frame
            event_frame = to_event_frame(
                rect[..., 0], rect[..., 1],
                bp,
                H=H, W=W
            )

            dt_ms = ( (bts[-1] - bts[0]) * 1e3 )

            yield event_frame, tss_imgs_us[img_i], dt_ms

        return

    # ==================================================================
    # CASE 2 — FAST TEMPORAL SLICING USING BINARY SEARCH
    # ==================================================================
    print(f"[MVSEC] Using FAST temporal slicing: dT_ms={dT_ms}")

    ts_us = (ts * 1e6).astype(np.int64)
    dt_us = int(dT_ms * 1000)

    ts_start = ts_us[0]
    ts_end   = ts_us[-1]

    t0_list = np.arange(ts_start, ts_end, dt_us)

    for t0 in t0_list:

        t1 = t0 + dt_us

        start = np.searchsorted(ts_us, t0, side="left")
        end   = np.searchsorted(ts_us, t1, side="left")

        if end <= start:
            continue

        bx  = x[start:end]
        by  = y[start:end]
        bp  = pol[start:end]
        bts = ts[start:end]

        if not rectify:
            rect = np.stack([bx, by], axis=-1)
        else:
            rect = rectify_map[
                by.astype(np.int32),
                bx.astype(np.int32)
            ]

            
        event_frame = to_event_frame(
            rect[..., 0], rect[..., 1],
            bp,
            H=H, W=W
        )

        dt_ms = ( (bts[-1] - bts[0]) * 1e3 )

        yield event_frame, t0, dt_ms



# ======================================================================
#   ADAPTIVE SLICING (NO-ALL_EVS VERSION)
# ======================================================================
def mvsec_evs_iterator_adaptive(scenedir, side="left", H=260, W=346, dt_function=None):

    print(f"[MVSEC] Adaptive slicing enabled")

    h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
    datain = h5py.File(h5in[0], 'r')

    evs = datain["davis"][side]["events"]

    # Separate columns
    x  = evs[:, 0].astype(np.uint16)
    y  = evs[:, 1].astype(np.uint16)
    ts = evs[:, 2].astype(np.float64)
    pol = evs[:, 3].astype(np.int8)

    rectify_map = read_rmap(
        os.path.join(scenedir, f"rectify_map_{side}.h5"),
        H=H, W=W
    )

    datain.close()

    ts_us = (ts * 1e6).astype(np.int64)

    t0 = ts_us[0]
    ts_end = ts_us[-1]
    iteration = 0

    # Sliding window
    while t0 < ts_end:

        dt_ms = dt_function(t0, iteration)
        dt_us = int(dt_ms * 1000)
        t1 = t0 + dt_us

        start = np.searchsorted(ts_us, t0, side="left")
        end   = np.searchsorted(ts_us, t1, side="left")

        if end > start:

            bx  = x[start:end]
            by  = y[start:end]
            bp  = pol[start:end]

            rect = rectify_map[
                by.astype(np.int32),
                bx.astype(np.int32)
            ]

            event_frame = to_event_frame(
                rect[..., 0], rect[..., 1],
                bp,
                H=H, W=W
            )

            yield event_frame, t0, dt_ms

        t0 = t1
        iteration += 1
