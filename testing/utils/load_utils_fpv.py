import os
import glob
import bisect

import h5py
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt  # kept because you had it
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2

from testing.utils.event_utils import to_event_frame


# ============================================================
# Defaults
# ============================================================

DEFAULT_MAX_EVENTS_LOADED = 1_000_000
DEFAULT_BATCH_EVENTS = 2_000_000

# valid timestamp ranges for each DVX scene (in microseconds). debugging purposes
VALID_FRAME_RANGES = {
    "day1_events": (1724173779479876, 1724173860879876),
    "day2_events": (1724175809154063, 1724175902754063),
    "day3_events": (1724314265805073, 1724314359805073),
}

# # valid timestamp ranges for each DVX scene (in microseconds). OG
# VALID_FRAME_RANGES = {
#     "day1_events": (1724173779479876, 1724173860879876),
#     "day2_events": (1724175809154063, 1724175902754063),
#     "day3_events": (1724314265805073, 1724314359805073),
# }



def get_valid_range_from_scene(scenedir):
    scenedir_lower = scenedir.lower()
    for key, rng in VALID_FRAME_RANGES.items():
        if key in scenedir_lower:
            print(f"[DVX] Auto valid timestamp range for {key}: {rng}")
            return rng
    print("[DVX] No valid timestamp range defined for this scene.")
    return None


# ============================================================
# Helper: slice a time-indexed HDF5 group by [t0_us, t1_us)
# ============================================================
def slice_group_by_time(group, ts_array, t0_us, t1_us, ts_key_name):
    """
    group:     h5py Group with multiple datasets
    ts_array:  np.ndarray of timestamps (same as group[ts_key_name][:])
    t0_us, t1_us: microsecond window [t0_us, t1_us)
    ts_key_name: name of timestamp dataset inside group ("timestamp" or "Timestamp")

    Returns:
      dict: { dataset_name: np.ndarray[slice] } for all datasets in group.
      If no samples in range, returns an empty dict.
    """
        
    if ts_array is None or len(ts_array) == 0:
        return {}

    start_idx = np.searchsorted(ts_array, t0_us, side="left")
    end_idx   = np.searchsorted(ts_array, t1_us, side="left")

    if end_idx <= start_idx:
        return {}

    return {
        name: ds[start_idx:end_idx]
        for name, ds in group.items()
    }


# ============================================================
# UZH-FPV iterator — rewritten with SIMPLE slicing logic
# ============================================================
def fpv_evs_iterator(
    scenedir,
    slicing_type="fixed",
    dT_ms=10.0,
    adaptive_slicer_pid=None,
    H=260, W=346,
    use_valid_frame_range=False,
    batch_events=DEFAULT_BATCH_EVENTS,
):
    """
    Streaming iterator over UZH-FPV HDF5 data.

    Yields, for each time slice, a dict:

        {
          "event_frame":  (H x W) np.ndarray (uint8/float),
          "t0_us":        int, start timestamp (microseconds),
          "t1_us":        int, end timestamp (microseconds),
          "dt_ms":        float, actual dt for this slice in ms,

          "cam_imu":       dict of np.ndarrays sliced in [t0_us, t1_us),
          "gt_state":      dict of np.ndarrays sliced in [t0_us, t1_us),
          "gt_cam_state":    dict of np.ndarrays sliced in [t0_us, t1_us),
        }

    Assumptions on HDF5 structure (created by your converter):

        /cam/events
            t   (N,) uint16
            x   (N,) uint16
            y   (N,) int64  (microseconds)
            p   (N,) int8   (polarity)

        /cam/imu
            timestamp (M,) int64
            timestamp, ax, ay, az, gx, gy, gz

        /gt/state
            timestamp (K,) int64
            timestamp, px, py, pz, qx, qy, qz, qw
        
        /gt_cam/state
            timestamp (K,) int64
            timestamp, px, py, pz, qx, qy, qz, qw
    """
        
    print(f"[UZH-FPV] Streaming from {scenedir}")

    valid_ts_range = get_valid_range_from_scene(scenedir) if use_valid_frame_range else None


    # ---------------------------------------------------------
    # Load HDF5 file
    # ---------------------------------------------------------
    h5_main = glob.glob(os.path.join(scenedir, "*.hdf5"))
    assert len(h5_main) == 1, "Expected exactly one HDF5 per scene"

    f_ev = h5py.File(h5_main[0], "r")
    evs = f_ev["cam/events"]

    x_ds = evs["x"]
    y_ds = evs["y"]
    t_ds = evs["t"]
    p_ds = evs["p"]
    N = x_ds.shape[0]

    print(f"[UZH-FPV] Total events: {N}")

    # ---------------- CAM IMU -------------------
    cam_imu_group = (
        f_ev["cam/imu"]
        if "cam" in f_ev and "imu" in f_ev["cam"]
        else None
    )
    cam_imu_ts = cam_imu_group["timestamp"][:] if cam_imu_group else None

    # ---------------- GT STATE -------------------
    gt_state_group = (
        f_ev["gt/state"]
        if "gt" in f_ev and "state" in f_ev["gt"]
        else None
    )
    gt_state_ts = gt_state_group["timestamp"][:] if gt_state_group else None

    # ---------------- GT CAM STATE -------------------
    gt_cam_state_group = (
        f_ev["gt_cam/state"]
        if "gt_cam" in f_ev and "state" in f_ev["gt_cam"]
        else None
    )
    gt_cam_state_ts = gt_cam_state_group["timestamp"][:] if gt_cam_state_group else None


    # ---------------------------------------------------------
    # Streaming buffer (sliding)
    # ---------------------------------------------------------
    x_buf = np.empty(0, np.uint16)
    y_buf = np.empty(0, np.uint16)
    t_buf = np.empty(0, np.int64)
    p_buf = np.empty(0, np.int8)

    load_cursor = 0

    # ---------------------------------------------------------
    # Load next chunk from HDF5
    # ---------------------------------------------------------
    def load_next_batch():
        nonlocal load_cursor, x_buf, y_buf, t_buf, p_buf

        if load_cursor >= N:
            return False

        end = min(load_cursor + batch_events, N)

        x = x_ds[load_cursor:end]
        y = y_ds[load_cursor:end]
        t = t_ds[load_cursor:end]
        p = p_ds[load_cursor:end]

        x_buf = np.concatenate([x_buf, x])
        y_buf = np.concatenate([y_buf, y])
        t_buf = np.concatenate([t_buf, t])
        p_buf = np.concatenate([p_buf, p])

        load_cursor = end
        return True

    # Initial load
    load_next_batch()

# ----------------------------
# HANDLE VALID TIMESTAMP RANGE
# ----------------------------
    if valid_ts_range is not None:
        start_us, end_us = valid_ts_range
        print(f"[UZH-FPV] Using valid timestamp window: {start_us} → {end_us}")

        # Ensure buffer covers the beginning of valid window
        while len(t_buf) == 0 or t_buf[-1] < start_us:
            if not load_next_batch():
                print("[UZH-FPV] No events found in valid range!")
                f_ev.close()
                return

        # Drop all events before start_us
        drop_idx = np.searchsorted(t_buf, start_us, side="left")
        x_buf = x_buf[drop_idx:]
        y_buf = y_buf[drop_idx:]
        t_buf = t_buf[drop_idx:]
        p_buf = p_buf[drop_idx:]

        # Set slicing start exactly at the valid window start
        t0_us = start_us
    else:
        # Start from very first event
        t0_us = t_buf[0]

    t_end_us = end_us if valid_ts_range is not None else t_ds[-1]
    current_dt_ms = float(dT_ms)


    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    while t0_us < t_end_us:

        t1_us = t0_us + int(current_dt_ms * 1000)

        # -----------------------------------------------------
        # Ensure buffer covers t1_us. If not, keep loading.
        # -----------------------------------------------------
        while len(t_buf) == 0 or t_buf[-1] < t1_us:
            if not load_next_batch():
                break  # no more events

        if len(t_buf) == 0:
            break

        # -----------------------------------------------------
        # Find slice indices inside current buffer
        # -----------------------------------------------------
        start_idx = np.searchsorted(t_buf, t0_us, side="left")
        end_idx   = np.searchsorted(t_buf, t1_us, side="left")

        # If no events fall in this window -> skip
        if end_idx <= start_idx:
            t0_us = t1_us
            continue

        # -----------------------------------------------------
        # Extract events
        # -----------------------------------------------------
        bx = x_buf[start_idx:end_idx]
        by = y_buf[start_idx:end_idx]
        bp = p_buf[start_idx:end_idx]

        # Build event frame
        event_frame = to_event_frame(bx, by, bp, H, W)

        # Actual dt measurement
        dt_ms_here = (t_buf[end_idx - 1] - t_buf[start_idx]) * 1e-3

        # -----------------------------------------------------
        # Slice IMU / PX4 data
        # -----------------------------------------------------
        cam_imu_slice = {}
        if cam_imu_group is not None:
            cam_imu_slice = slice_group_by_time(
                cam_imu_group, cam_imu_ts, t0_us, t1_us, "timestamp"
            )

        gt_state_slice = {}
        if gt_state_group is not None:
            gt_state_slice = slice_group_by_time(
                gt_state_group, gt_state_ts, t0_us, t1_us, "timestamp"
            )
        
        px4_state_slice = {}
        if gt_cam_state_group is not None:
            px4_state_slice = slice_group_by_time(
                gt_cam_state_group, gt_cam_state_ts, t0_us, t1_us, "timestamp"
            )
        

        # -----------------------------------------------------
        # Yield slice
        # -----------------------------------------------------
        yield {
            "event_frame": event_frame,
            "t0_us": int(t0_us),
            "t1_us": int(t1_us),
            "dt_ms": float(dt_ms_here),
            "cam_imu": cam_imu_slice,
            "gt_state": gt_state_slice,
            "px4_state": px4_state_slice,
        }

        # -----------------------------------------------------
        # Adaptive slicing update
        # -----------------------------------------------------
        if slicing_type == "adaptive" and adaptive_slicer_pid is not None:
            try:
                n_events = end_idx - start_idx
                new_dt = adaptive_slicer_pid.update(current_dt_ms, n_events)
                if new_dt is not None and new_dt > 0:
                    current_dt_ms = float(new_dt)
            except Exception as e:
                print("[WARN] PID update failed:", e)

        # -----------------------------------------------------
        # REMOVE CONSUMED BUFFER PART
        # -----------------------------------------------------
        x_buf = x_buf[end_idx:]
        y_buf = y_buf[end_idx:]
        t_buf = t_buf[end_idx:]
        p_buf = p_buf[end_idx:]

        # -----------------------------------------------------
        # Next slice
        # -----------------------------------------------------
        t0_us = t1_us

    f_ev.close()
    print("[UZH-FPV] Done")
