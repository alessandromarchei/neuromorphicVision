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

  
  frames dt_ms defaults :
    outdoor: 21.941 ms → 45 hz
    indoor: 31.859 ms → ~31 hz
  
  flow_gt dt_ms defaults :
    outdoor: 50.0 ms → ~20 hz
    indoor: 50.0 ms → ~20 hz

    
    LOGIC : 
    20hz : means use the normal gt flow frequency
    dt=1 : means use the gt flow upsampled to the APS camera frequency (45 hz for outdoor, 31 hz for indoor)
    dt=4 : means use the gt flow downsampled to 1/4 of the APS camera frequency (11 hz for outdoor, 8 hz for indoor)
  events_idxs : 
  - outdoor day1: starts at -1
  - indoor flying : starts at N > 0
Total ≈ 13 bytes/event instead of 32 bytes/event.
"""

# Default limits (can be overridden in function params)
DEFAULT_MAX_EVENTS_LOADED = 1_000_000
DEFAULT_BATCH_EVENTS = 200_000

def read_rmap(rect_file, H=180, W=240):
    h5file = glob.glob(rect_file)[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    assert rectify_map.shape == (H, W, 2)
    rmap.close()
    return rectify_map


def mvsec_evs_iterator(
    scenedir,
    side="left",
    dT_ms=None,
    H=260,
    W=346,
    rectify=True,
    gt_mode="20hz",  # NEW: "20hz" | "dt1" | "dt4"
    max_events_loaded=DEFAULT_MAX_EVENTS_LOADED,
    batch_events=DEFAULT_BATCH_EVENTS,
):
    """
    Streaming iterator that yields tuples:

        event_frame,
        t_event_us,
        dt_ms,
        flow_map,
        flow_ts_us,
        flow_dt_ms

    gt_mode:
        - "20hz" → *_gt.hdf5
        - "dt1"  → new scene.h5 → flow/dt=1
        - "dt4"  → new scene.h5 → flow/dt=4
    """

    print(f"[MVSEC] Streaming from {scenedir}, side={side}, dT_ms={dT_ms}, GT={gt_mode}")

    # ---------------------------------------------------------
    # Load EVENTS from *_data.hdf5
    # ---------------------------------------------------------
    h5_main = glob.glob(os.path.join(scenedir, "*_data.hdf5"))
    assert len(h5_main) == 1
    f_ev = h5py.File(h5_main[0], "r")
    evs = f_ev[f"davis/{side}/events"]
    N = evs.shape[0]

    rectify_map = read_rmap(os.path.join(scenedir, f"rectify_map_{side}.h5"), H=H, W=W)

    # ---------------------------------------------------------
    # SELECT FLOW SOURCE
    # ---------------------------------------------------------
    flow_dset = None
    flow_ts_us = None

    if gt_mode == "20hz":
        print("[GT] Using *_gt.hdf5 (20 Hz)")
        h5_gt = glob.glob(os.path.join(scenedir, "*_gt.hdf5"))
        assert len(h5_gt) == 1
        f_gt = h5py.File(h5_gt[0], "r")
        flow_dset = f_gt[f"davis/{side}/flow_dist"]
        ts_sec = f_gt[f"davis/{side}/flow_dist_ts"][:]
        flow_ts_us = (ts_sec * 1e6).astype(np.int64)

    elif gt_mode in ("dt1", "dt4"):
        dt = 1 if gt_mode == "dt1" else 4
        print(f"[GT] Using {scenedir}*dt.h5 → flow/dt={dt}")

        h5_dt = glob.glob(os.path.join(scenedir, "*dt.h5"))
        assert len(h5_dt) == 1
        f_gt = h5py.File(h5_dt[0], "r")

        group = f_gt[f"flow/dt={dt}"]
        flow_dset = [k for k in group.keys() if k != "timestamps"]
        flow_dset.sort()
        flow_dset = [group[k] for k in flow_dset]

        ts_pairs = group["timestamps"][:]   # (N, 2) prev_t, cur_t
        flow_ts_us = (ts_pairs[:, 1] * 1e6).astype(np.int64)

        f_gt_dt = f_gt  # for closing later

    else:
        raise ValueError("gt_mode must be: '20hz', 'dt1', 'dt4'")

    print(f"[GT] Loaded {len(flow_ts_us)} flow timestamps")

    # Prevent OOM by streaming GT only when needed
    def load_gt_for(t_us):
        """Return nearest GT flow lazily."""
        idx = np.searchsorted(flow_ts_us, t_us, side="left")
        if idx == 0:
            best = 0
        elif idx >= len(flow_ts_us):
            best = len(flow_ts_us) - 1
        else:
            before = idx - 1
            after = idx
            best = before if abs(flow_ts_us[before] - t_us) <= abs(flow_ts_us[after] - t_us) else after

        # Load flow map
        if gt_mode == "20hz":
            flow_map = flow_dset[best]     # (2,H,W)
        else:
            flow_map = flow_dset[best][...]  # dataset in list

        # Compute dt_ms
        dt_ms = None
        if best > 0:
            dt_ms = (flow_ts_us[best] - flow_ts_us[best - 1]) * 1e-3

        flow_id = best


        return flow_map, flow_ts_us[best], dt_ms, flow_id


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
        """Load next event batch"""
        nonlocal load_cursor, x_buf, y_buf, ts_us_buf, pol_buf
        if load_cursor >= N:
            return False
        end = min(load_cursor + batch_events, N)
        x = evs[load_cursor:end, 0].astype(np.uint16)
        y = evs[load_cursor:end, 1].astype(np.uint16)
        ts = evs[load_cursor:end, 2].astype(np.float64)
        p  = evs[load_cursor:end, 3].astype(np.int8)
        x_buf = np.concatenate([x_buf, x])
        y_buf = np.concatenate([y_buf, y])
        ts_us_buf = np.concatenate([ts_us_buf, (ts * 1e6).astype(np.int64)])
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
    # CASE A: MVSEC IMAGE SLICING (dT_ms=None)
    # =======================================================================
    if dT_ms is None:
        print("[MVSEC] Using original MVSEC slicing (APS timestamps).")

        idxs = f_ev[f"davis/{side}/image_raw_event_inds"][:]
        img_ts = f_ev[f"davis/{side}/image_raw_ts"][:]
        img_ts_us = (img_ts * 1e6).astype(np.int64)

        if idxs[0] == -1:  # OutdoorDay1 weird indexing
            idxs = idxs[1:]
            img_ts_us = img_ts_us[1:]

        prev_end = 0

        for i, ts_us in enumerate(img_ts_us):
            idx1 = int(idxs[i])
            idx0 = prev_end
            prev_end = idx1

            # Fill buffer
            while abs_start + len(ts_us_buf) < idx1:
                if not load_more():
                    break
                trim_buffer()

            local0 = idx0 - abs_start
            local1 = idx1 - abs_start

            bx = x_buf[local0:local1]
            by = y_buf[local0:local1]
            bp = pol_buf[local0:local1]

            rect = rectify_map[by.astype(np.int32), bx.astype(np.int32)]
            xs, ys = rect[:, 0], rect[:, 1]

            event_frame = to_event_frame(xs, ys, bp, H, W)
            dt_ms_here = (ts_us_buf[local1 - 1] - ts_us_buf[local0]) * 1e-3

            # GT sample nearest to this image_ts
            flow_map, flow_ts, flow_dt, flow_id = load_gt_for(ts_us)

            #reshape the output flow to eventually 2, H, W
            if flow_map.shape[2] == 2:
                flow_map = flow_map.transpose(2, 0, 1)
            
            flow_map = mask_outdoor_carhood(flow_map)


            yield event_frame, ts_us, dt_ms_here, flow_map, flow_ts, flow_dt, flow_id

        f_ev.close()
        return


    # =======================================================================
    # CASE B: FIXED TEMPORAL SLICING (dT_ms > 0)
    # =======================================================================
    print(f"[MVSEC] Fixed slicing: every {dT_ms} ms")

    if load_cursor == 0:
        load_more()

    dt_us = int(dT_ms * 1000)
    t0_us = ts_us_buf[0]
    t_end_us = int(float(evs[-1, 2]) * 1e6)

    while t0_us < t_end_us:
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

            rect = rectify_map[by.astype(np.int32), bx.astype(np.int32)]
            xs, ys = rect[:, 0], rect[:, 1]
            event_frame = to_event_frame(xs, ys, bp, H, W)

            # GT nearest to t0_us
            flow_map, flow_ts, flow_dt, flow_id = load_gt_for(t0_us)

            dt_ms_here = (ts_us_buf[end - 1] - ts_us_buf[start]) * 1e-3

            #reshape the output flow to eventually 2, H, W
            if flow_map.shape[2] == 2:
                flow_map = flow_map.transpose(2, 0, 1)

            flow_map = mask_outdoor_carhood(flow_map)
            
            yield event_frame, t0_us, dt_ms_here, flow_map, flow_ts, flow_dt, flow_id

        t0_us = t1_us
        trim_buffer()

    f_ev.close()

def mvsec_evs_iterator_adaptive(
    scenedir,
    side,
    adaptive_slicer,          # object providing get_current_dt_ms()
    H=260,
    W=346,
    rectify=True,
    gt_mode="20hz",           # NEW: "20hz", "dt1", "dt4"
    max_events_loaded=DEFAULT_MAX_EVENTS_LOADED,
    batch_events=DEFAULT_BATCH_EVENTS,
):
    """
    Adaptive slicing MVSEC iterator.
    Supports GT at:
        - 20Hz  (original MVSEC flow_dist)
        - dt=1  (upsampled flow)
        - dt=4  (downsampled flow)
    
    Yields:
        event_frame,
        t_us,
        dt_ms,
        flow_map,
        ts_gt_us,
        dt_gt_ms
    """

    print(f"[MVSEC-ADAPTIVE] Streaming from {scenedir}, GT={gt_mode}")

    # ---------------------------------------------------------
    # Load main event HDF5
    # ---------------------------------------------------------
    h5_main = glob.glob(os.path.join(scenedir, "*_data.hdf5"))
    assert len(h5_main) == 1
    f_ev = h5py.File(h5_main[0], "r")

    evs = f_ev[f"davis/{side}/events"]
    N = evs.shape[0]

    rectify_map = read_rmap(os.path.join(scenedir, f"rectify_map_{side}.h5"), H=H, W=W)

    # =====================================================================
    # SELECT WHICH GT WE USE  (20Hz, dt=1, or dt=4)
    # =====================================================================
    flow_dset = None
    flow_ts_us = None
    dt_mode = None

    if gt_mode == "20hz":
        print("[GT] Using *_gt.hdf5 (20Hz)")
        h5_gt = glob.glob(os.path.join(scenedir, "*_gt.hdf5"))
        assert len(h5_gt) == 1
        f_gt = h5py.File(h5_gt[0], "r")

        flow_dset = f_gt[f"davis/{side}/flow_dist"]  # shape (Ngt, 2, H, W)
        ts_sec = f_gt[f"davis/{side}/flow_dist_ts"][:]
        flow_ts_us = (ts_sec * 1e6).astype(np.int64)
        dt_mode = "20hz"

    elif gt_mode in ("dt1", "dt4"):
        dt = 1 if gt_mode == "dt1" else 4
        dt_mode = f"dt={dt}"
        print(f"[GT] Using scene.h5 → flow/{dt_mode}")

        h5_dt = glob.glob(os.path.join(scenedir, "*.h5"))
        assert len(h5_dt) == 1
        f_gt = h5py.File(h5_dt[0], "r")

        group = f_gt[f"flow/{dt_mode}"]

        # sorted list of datasets (flow_00000000, flow_00000001, ...)
        flow_keys = sorted([k for k in group.keys() if k != "timestamps"])
        flow_dset = [group[k] for k in flow_keys]

        # timestamps: (N,2) → prev_t, cur_t
        ts_pairs = group["timestamps"][:]
        flow_ts_us = (ts_pairs[:, 1] * 1e6).astype(np.int64)  # use end timestamp as ground-truth time

    else:
        raise ValueError("gt_mode must be '20hz', 'dt1', or 'dt4'")

    print(f"[GT] Loaded {len(flow_ts_us)} flow timestamps.")


    # ---------------------------------------------------------
    # Function to load GT lazily (without RAM explosion)
    # ---------------------------------------------------------
    def load_gt_for(t_us):
        """Return nearest GT flow and its timestamp."""
        idx = np.searchsorted(flow_ts_us, t_us, side="left")

        if idx == 0:
            best = 0
        elif idx >= len(flow_ts_us):
            best = len(flow_ts_us) - 1
        else:
            b = idx - 1
            a = idx
            best = b if abs(flow_ts_us[b] - t_us) <= abs(flow_ts_us[a] - t_us) else a

        # Load only the chosen map
        if dt_mode == "20hz":
            flow_map = flow_dset[best]           # (2,H,W)
        else:
            flow_map = flow_dset[best][...]     # dataset object to array

        # Compute dt between GT samples
        dt_gt_ms = None
        if best > 0:
            dt_gt_ms = (flow_ts_us[best] - flow_ts_us[best - 1]) * 1e-3
        
        flow_id = best

        return flow_map, flow_ts_us[best], dt_gt_ms, flow_id


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
        """Load next chunk of events."""
        nonlocal load_cursor, x_buf, y_buf, ts_us_buf, pol_buf
        if load_cursor >= N:
            return False

        end = min(load_cursor + batch_events, N)
        batch = evs[load_cursor:end]

        x = batch[:, 0].astype(np.uint16)
        y = batch[:, 1].astype(np.uint16)
        ts_us = (batch[:, 2] * 1e6).astype(np.int64)
        p = batch[:, 3].astype(np.int8)

        x_buf = np.concatenate([x_buf, x])
        y_buf = np.concatenate([y_buf, y])
        ts_us_buf = np.concatenate([ts_us_buf, ts_us])
        pol_buf = np.concatenate([pol_buf, p])

        load_cursor = end
        return True

    def trim_buffer():
        """Trim old events once buffer exceeds limit."""
        nonlocal abs_start, x_buf, y_buf, ts_us_buf, pol_buf
        extra = len(x_buf) - max_events_loaded
        if extra > 0:
            x_buf = x_buf[extra:]
            y_buf = y_buf[extra:]
            ts_us_buf = ts_us_buf[extra:]
            pol_buf = pol_buf[extra:]
            abs_start += extra


    # ---------------------------------------------------------
    # INITIAL LOAD
    # ---------------------------------------------------------
    if load_cursor == 0:
        load_more()

    t_end_us = int(float(evs[-1, 2]) * 1e6)
    t0_us = ts_us_buf[0]


    # ================================================================
    #                 ADAPTIVE SLICING LOOP
    # ================================================================
    print("[MVSEC-ADAPTIVE] Starting adaptive loop...")

    while t0_us < t_end_us:

        # 1) get dt from adaptive controller
        dt_ms = adaptive_slicer.get_current_dt_ms()
        dt_us = int(dt_ms * 1000)
        t1_us = t0_us + dt_us

        # 2) ensure buffer covers t1_us
        while len(ts_us_buf) == 0 or ts_us_buf[-1] < t1_us:
            if not load_more():
                break
            trim_buffer()

        if len(ts_us_buf) == 0:
            break

        # 3) find event slice within buffer
        start = np.searchsorted(ts_us_buf, t0_us, side="left")
        end   = np.searchsorted(ts_us_buf, t1_us, side="left")

        if end > start:
            bx = x_buf[start:end]
            by = y_buf[start:end]
            bp = pol_buf[start:end]

            # rectification
            if rectify:
                rect = rectify_map[by.astype(np.int32), bx.astype(np.int32)]
                xs = rect[:, 0]
                ys = rect[:, 1]
            else:
                xs, ys = bx, by

            event_frame = to_event_frame(xs, ys, bp, H, W)

            # 4) load GT lazily
            flow_map, ts_gt_us, gt_dt_ms, flow_id = load_gt_for(t0_us)

            #reshape the output flow to eventually 2, H, W
            if flow_map.shape[2] == 2:
                flow_map = flow_map.transpose(2, 0, 1)

            flow_map = mask_outdoor_carhood(flow_map)

            # 5) yield sample
            yield event_frame, t0_us, dt_ms, flow_map, ts_gt_us, gt_dt_ms, flow_id

        # 6) next window
        t0_us = t1_us
        trim_buffer()

    f_ev.close()


def mask_outdoor_carhood(flow_map):
    """
    Removes the car-hood region for MVSEC outdoor sequences.
    flow_map must be (2, H, W).
    """
    if flow_map.ndim == 3 and flow_map.shape[0] == 2:
        flow_map[:, 193:, :] = 0
    return flow_map
