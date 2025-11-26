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

  
  dt_ms defaults :
  outdoor: 20 ms → 50 hz
  indoor: 30 ms → ~33 hz
  
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
    max_events_loaded=DEFAULT_MAX_EVENTS_LOADED,
    batch_events=DEFAULT_BATCH_EVENTS,
):
    """
    Streaming iterator returning:
        event_frame,
        t_event_us,
        event_dt_ms,
        gt_flow_map,
        gt_flow_ts_us,
        gt_flow_dt_ms

    Supports:
        - MVSEC default slicing (dT_ms=None)
        - Fixed temporal slicing (dT_ms > 0)

    Loads:
        ✓ events in small batches
        ✓ 1 optical flow map at a time (closest in time)
    """

    print(f"[MVSEC] Streaming events+GT from {scenedir}, side={side}, dT_ms={dT_ms}")

    # ---------------------------------------------------------
    # Load main event HDF5
    # ---------------------------------------------------------
    h5_main = glob.glob(os.path.join(scenedir, "*_data.hdf5"))
    assert len(h5_main) == 1
    f_ev = h5py.File(h5_main[0], "r")
    evs = f_ev[f"davis/{side}/events"]
    N = evs.shape[0]

    # Rectification map
    rectify_map = read_rmap(os.path.join(scenedir, f"rectify_map_{side}.h5"), H=H, W=W)

    # ---------------------------------------------------------
    # Optional ground-truth flow (if available)
    # ---------------------------------------------------------
    h5_gt = glob.glob(os.path.join(scenedir, "*_gt.hdf5"))
    use_gt = len(h5_gt) > 0

    if use_gt:
        f_gt = h5py.File(h5_gt[0], "r")
        flow_dset = f_gt[f"davis/{side}/flow_dist"]        # (N_gt, 2, H, W)
        flow_ts   = f_gt[f"davis/{side}/flow_dist_ts"][:] # seconds
        flow_ts_us = (flow_ts * 1e6).astype(np.int64)
        print(f"[MVSEC] GT flow loaded lazily, N_gt={len(flow_ts_us)}")
    else:
        flow_dset = None
        flow_ts_us = None
        print("[MVSEC] No GT flow file found.")

    # ---------------------------------------------------------
    # Event buffer for streaming
    # ---------------------------------------------------------
    x_buf = np.empty(0, dtype=np.uint16)
    y_buf = np.empty(0, dtype=np.uint16)
    ts_us_buf = np.empty(0, dtype=np.int64)
    pol_buf = np.empty(0, dtype=np.int8)

    abs_start = 0
    load_cursor = 0

    def load_more():
        nonlocal load_cursor, x_buf, y_buf, ts_us_buf, pol_buf
        if load_cursor >= N:
            return False
        end = min(load_cursor + batch_events, N)

        x = evs[load_cursor:end, 0].astype(np.uint16)
        y = evs[load_cursor:end, 1].astype(np.uint16)
        ts = evs[load_cursor:end, 2].astype(np.float64)
        p  = evs[load_cursor:end, 3].astype(np.int8)

        ts_us = (ts * 1e6).astype(np.int64)

        x_buf = np.concatenate([x_buf, x])
        y_buf = np.concatenate([y_buf, y])
        ts_us_buf = np.concatenate([ts_us_buf, ts_us])
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

    # ---------------------------------------------------------
    # Helper: load GT closest to timestamp t_us
    # ---------------------------------------------------------
    def load_gt_for(t_us):
        if not use_gt:
            return None, None, None

        idx = np.searchsorted(flow_ts_us, t_us, side="left")

        # choose nearest
        if idx == 0:
            best = 0
        elif idx == len(flow_ts_us):
            best = idx - 1
        else:
            before = idx - 1
            after  = idx
            if abs(flow_ts_us[before] - t_us) <= abs(flow_ts_us[after] - t_us):
                best = before
            else:
                best = after

        flow_map = flow_dset[best]           # loaded on demand
        ts_gt = flow_ts_us[best]
        dt_gt_ms = None
        if best > 0:
            dt_gt_ms = (flow_ts_us[best] - flow_ts_us[best - 1]) * 1e-3

        return flow_map, ts_gt, dt_gt_ms

    # ---------------------------------------------------------
    # CASE A: MVSEC image slicing
    # ---------------------------------------------------------
    if dT_ms is None:
        print("[MVSEC] Using MVSEC image slicing.")
        event_idxs = f_ev[f"davis/{side}/image_raw_event_inds"][:]
        tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"tss_imgs_us_{side}.txt")))

        #event idxs starts at -1 for outdoor day1 -> prevent error by removing the first index
        if event_idxs[0] == -1:
            event_idxs = event_idxs[1:]
            tss_imgs_us = tss_imgs_us[1:]

        prev_end = 0
        for img_i, img_ts_us in enumerate(tss_imgs_us):

            idx1 = int(event_idxs[img_i])
            idx0 = prev_end
            prev_end = idx1

            while abs_start + len(x_buf) < idx1:
                if not load_more():
                    break
                trim_buffer()

            local0 = idx0 - abs_start
            local1 = idx1 - abs_start
            if local0 < 0 or local1 > len(x_buf):
                raise RuntimeError("Buffer does not contain required event range.")

            bx = x_buf[local0:local1]
            by = y_buf[local0:local1]
            bts_us = ts_us_buf[local0:local1]
            bp = pol_buf[local0:local1]

            if rectify:
                rect = rectify_map[by.astype(np.int32), bx.astype(np.int32)]
                xs = rect[:,0]
                ys = rect[:,1]
            else:
                xs = bx
                ys = by

            event_frame = to_event_frame(xs, ys, bp, H, W)
            dt_ms = (bts_us[-1] - bts_us[0]) * 1e-3

            # load GT
            flow_map, ts_gt, dt_gt_ms = load_gt_for(img_ts_us)

            yield event_frame, img_ts_us, dt_ms, flow_map, ts_gt, dt_gt_ms

        f_ev.close()
        if use_gt:
            f_gt.close()
        return

    # ---------------------------------------------------------
    # CASE B: fixed temporal slicing
    # ---------------------------------------------------------
    print("[MVSEC] Using fixed temporal slicing.")
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

        if len(ts_us_buf) == 0:
            break

        start = np.searchsorted(ts_us_buf, t0_us, side="left")
        end   = np.searchsorted(ts_us_buf, t1_us, side="left")

        if end > start:
            bx = x_buf[start:end]
            by = y_buf[start:end]
            bp = pol_buf[start:end]
            bts_us = ts_us_buf[start:end]

            if rectify:
                rect = rectify_map[by.astype(np.int32), bx.astype(np.int32)]
                xs = rect[:,0]
                ys = rect[:,1]
            else:
                xs = bx
                ys = by

            event_frame = to_event_frame(xs, ys, bp, H, W)
            dt_ms_batch = (bts_us[-1] - bts_us[0]) * 1e-3

            # GT
            flow_map, ts_gt, dt_gt_ms = load_gt_for(t0_us)

            yield event_frame, t0_us, dt_ms_batch, flow_map, ts_gt, dt_gt_ms

        t0_us = t1_us
        trim_buffer()

    f_ev.close()
    if use_gt:
        f_gt.close()


# ======================================================================
#   ADAPTIVE SLICING (STREAMING, NO ALL_EVS)
# ======================================================================
def mvsec_evs_iterator_adaptive(
    scenedir,
    side="left",
    H=260,
    W=346,
    dt_function=None,
    rectify=True,
    max_events_loaded=DEFAULT_MAX_EVENTS_LOADED,
    batch_events=DEFAULT_BATCH_EVENTS,
):
    """
    Adaptive temporal slicing.

    dt_function(t0_us, iteration) -> dt_ms for the current window.

    t0_us is the start time (in microseconds) of the current window,
    and `iteration` is 0, 1, 2, ...

    Yields: event_frame, t0_us, dt_ms
    """

    print(f"[MVSEC] Adaptive slicing enabled (side={side})")
    print(f"[MVSEC] max_events_loaded={max_events_loaded}, batch_events={batch_events}")

    h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
    assert len(h5in) == 1

    datain = h5py.File(h5in[0], "r")
    evs = datain["davis"][side]["events"]
    total_events = evs.shape[0]

    rectify_map = read_rmap(os.path.join(scenedir, f"rectify_map_{side}.h5"), H=H, W=W)

    print(f"[MVSEC] Total events: {total_events:,}")

    try:
        # Buffer state
        x_buf = np.empty(0, dtype=np.uint16)
        y_buf = np.empty(0, dtype=np.uint16)
        ts_buf = np.empty(0, dtype=np.float64)
        ts_us_buf = np.empty(0, dtype=np.int64)
        pol_buf = np.empty(0, dtype=np.int8)

        buf_start_idx_abs = 0
        load_cursor_abs = 0

        last_ts_sec = float(evs[-1, 2])
        t_end_us = int(last_ts_sec * 1e6)

        def load_more_events_time():
            nonlocal x_buf, y_buf, ts_buf, ts_us_buf, pol_buf
            nonlocal load_cursor_abs, buf_start_idx_abs

            if load_cursor_abs >= total_events:
                return

            end = min(load_cursor_abs + batch_events, total_events)

            x_chunk = evs[load_cursor_abs:end, 0].astype(np.uint16)
            y_chunk = evs[load_cursor_abs:end, 1].astype(np.uint16)
            ts_chunk = evs[load_cursor_abs:end, 2].astype(np.float64)
            pol_chunk = evs[load_cursor_abs:end, 3].astype(np.int8)
            ts_us_chunk = (ts_chunk * 1e6).astype(np.int64)

            if x_buf.size == 0:
                x_buf = x_chunk
                y_buf = y_chunk
                ts_buf = ts_chunk
                ts_us_buf = ts_us_chunk
                pol_buf = pol_chunk
            else:
                x_buf = np.concatenate([x_buf, x_chunk])
                y_buf = np.concatenate([y_buf, y_chunk])
                ts_buf = np.concatenate([ts_buf, ts_chunk])
                ts_us_buf = np.concatenate([ts_us_buf, ts_us_chunk])
                pol_buf = np.concatenate([pol_buf, pol_chunk])

            load_cursor_abs = end

        def ensure_buffer_covers_time(t1_us):
            nonlocal ts_us_buf

            while True:
                if ts_us_buf.size == 0:
                    if load_cursor_abs >= total_events:
                        break
                    load_more_events_time()
                    continue

                if ts_us_buf[-1] >= t1_us:
                    break

                if load_cursor_abs >= total_events:
                    break

                load_more_events_time()

        def trim_buffer_by_time(t0_us):
            nonlocal x_buf, y_buf, ts_buf, ts_us_buf, pol_buf, buf_start_idx_abs

            if ts_us_buf.size == 0:
                return

            cut = np.searchsorted(ts_us_buf, t0_us, side="left")
            if cut > 0:
                if cut >= ts_us_buf.size:
                    x_buf = np.empty(0, dtype=np.uint16)
                    y_buf = np.empty(0, dtype=np.uint16)
                    ts_buf = np.empty(0, dtype=np.float64)
                    ts_us_buf = np.empty(0, dtype=np.int64)
                    pol_buf = np.empty(0, dtype=np.int8)
                else:
                    x_buf = x_buf[cut:]
                    y_buf = y_buf[cut:]
                    ts_buf = ts_buf[cut:]
                    ts_us_buf = ts_us_buf[cut:]
                    pol_buf = pol_buf[cut:]

                buf_start_idx_abs += cut

        # Initialize
        load_more_events_time()
        if ts_us_buf.size == 0:
            return

        t0_us = int(ts_us_buf[0])
        iteration = 0

        while t0_us < t_end_us:

            if dt_function is None:
                raise ValueError("[MVSEC] dt_function must be provided for adaptive slicing.")

            dt_ms = float(dt_function(t0_us, iteration))
            dt_us = int(dt_ms * 1000)
            t1_us = t0_us + dt_us

            ensure_buffer_covers_time(t1_us)

            if ts_us_buf.size == 0:
                break

            start = np.searchsorted(ts_us_buf, t0_us, side="left")
            end = np.searchsorted(ts_us_buf, t1_us, side="left")

            if end > start:
                bx = x_buf[start:end]
                by = y_buf[start:end]
                bp = pol_buf[start:end]
                bts = ts_buf[start:end]

                if rectify:
                    rect = rectify_map[
                        by.astype(np.int32),
                        bx.astype(np.int32)
                    ]
                else:
                    rect = np.stack([bx, by], axis=-1)

                event_frame = to_event_frame(
                    rect[..., 0], rect[..., 1],
                    bp,
                    H=H, W=W
                )

                # dt_ms here is what we chose via dt_function, but we can also
                # compute the actual temporal span if you prefer:
                # dt_ms_actual = (bts[-1] - bts[0]) * 1e3

                yield event_frame, t0_us, dt_ms

            t0_us = t1_us
            iteration += 1

            trim_buffer_by_time(t0_us)

        return

    finally:
        datain.close()
