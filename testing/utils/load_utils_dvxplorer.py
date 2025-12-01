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

# valid timestamp ranges for each DVX scene (in microseconds)
VALID_FRAME_RANGES = {
    "day1_events": (1724173779479876, 1724173860879876),
    "day2_events": (1724175809154063, 1724175902754063),
    "day3_events": (1724314265805073, 1724314359805073),
}


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

    # find indices in timestamp array
    start_idx = np.searchsorted(ts_array, t0_us, side="left")
    end_idx = np.searchsorted(ts_array, t1_us, side="left")

    if end_idx <= start_idx:
        return {}

    out = {}
    for name, ds in group.items():
        # slice each dataset with [start_idx:end_idx]
        out[name] = ds[start_idx:end_idx]

    return out


# ============================================================
# Main DVXplorer iterator
# ============================================================

def dvxplorer_evs_iterator(
    scenedir,
    slicing_type="fixed",          # "fixed" or "adaptive"
    dT_ms=10.0,                    # initial dt in ms (used also as base for adaptive)
    adaptive_slicer_pid=None,      # object with method update(dt_ms, n_events) -> new_dt_ms (optional)
    H=480,
    W=640,
    max_events_loaded=DEFAULT_MAX_EVENTS_LOADED,
    batch_events=DEFAULT_BATCH_EVENTS,
    use_valid_frame_range=False,
):
    """
    Streaming iterator over DVXplorer HDF5 data.

    Yields, for each time slice, a dict:

        {
          "event_frame":  (H x W) np.ndarray (uint8/float),
          "t0_us":        int, start timestamp (microseconds),
          "t1_us":        int, end timestamp (microseconds),
          "dt_ms":        float, actual dt for this slice in ms,

          "dv_imu":       dict of np.ndarrays sliced in [t0_us, t1_us),
          "px4_imu":      dict of np.ndarrays sliced in [t0_us, t1_us),
          "px4_state":    dict of np.ndarrays sliced in [t0_us, t1_us),
        }

    Assumptions on HDF5 structure (created by your converter):

        /dvxplorermicro/events
            x   (N,) uint16
            y   (N,) uint16
            t   (N,) int64  (microseconds)
            p   (N,) int8   (polarity)

        /dvxplorermicro/imu
            timestamp (M,) int64
            ax, ay, az, gx, gy, gz, temperature ...

        /px4/imu
            Timestamp, qx, qy, qz, qw, gx, gy, gz, ax, ay, az ...

        /px4/state
            Timestamp, Lidar, DistanceGround, Airspeed, Groundspeed, RollAngle, PitchAngle, ...
    """

    print(f"[DVX] Streaming from {scenedir}, slicing_type={slicing_type}, dT_ms={dT_ms}")

    # 1) Valid range (in microseconds)
    valid_ts_range = get_valid_range_from_scene(scenedir) if use_valid_frame_range else None

    # 2) Open HDF5
    h5_main = glob.glob(os.path.join(scenedir, "*.hdf5"))
    assert len(h5_main) == 1, f"Expected exactly one HDF5 in {scenedir}, found {h5_main}"
    h5_path = h5_main[0]
    print(f"[DVX] Using HDF5 file: {h5_path}")

    f_ev = h5py.File(h5_path, "r")

    print("[HDF5] Top-level groups:", list(f_ev.keys()))
    for g in f_ev.keys():
        print(f"[HDF5] Group '{g}' contains:", list(f_ev[g].keys()))

    # 2.1) Events
    evs = f_ev["dvxplorermicro/events"]
    print("[HDF5] Events datasets:", list(evs.keys()))
    x_ds = evs["x"]
    y_ds = evs["y"]
    t_ds = evs["t"]
    p_ds = evs["p"]
    N = x_ds.shape[0]
    print(f"[DVX] N events: {N}")

    # 2.2) DV IMU (optional)
    dv_imu_group = f_ev["dvxplorermicro/imu"] if "dvxplorermicro" in f_ev and "imu" in f_ev["dvxplorermicro"] else None
    dv_imu_ts = dv_imu_group["timestamp"][:] if dv_imu_group is not None and "timestamp" in dv_imu_group else None

    # 2.3) PX4 IMU (optional)
    px4_group = f_ev["px4"] if "px4" in f_ev else None
    px4_imu_group = px4_group["imu"] if px4_group is not None and "imu" in px4_group else None
    px4_state_group = px4_group["state"] if px4_group is not None and "state" in px4_group else None

    px4_imu_ts = px4_imu_group["Timestamp"][:] if px4_imu_group is not None and "Timestamp" in px4_imu_group else None
    px4_state_ts = px4_state_group["Timestamp"][:] if px4_state_group is not None and "Timestamp" in px4_state_group else None

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

        x = x_ds[load_cursor:end].astype(np.uint16)
        y = y_ds[load_cursor:end].astype(np.uint16)
        ts = t_ds[load_cursor:end].astype(np.int64)   # microseconds
        p = p_ds[load_cursor:end].astype(np.int8)

        x_buf = np.concatenate([x_buf, x])
        y_buf = np.concatenate([y_buf, y])
        ts_us_buf = np.concatenate([ts_us_buf, ts])
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
    # SLICING
    # ---------------------------------------------------------
    if load_cursor == 0:
        # initial load
        if not load_more():
            print("[DVX] No events to load.")
            f_ev.close()
            return

    # current dt (ms) used for slicing; can be adapted
    current_dt_ms = float(dT_ms)
    print(f"[DVX] Initial dt_ms: {current_dt_ms}")

    t0_us = ts_us_buf[0]
    t_end_us = int(t_ds[-1])  # last event timestamp

    print(f"[DVX] Time span: [{t0_us}, {t_end_us}] us")

    # main slicing loop
    while t0_us < t_end_us:

        # optional valid range in timestamps
        if valid_ts_range is not None:
            start_us, end_us = valid_ts_range

            if t0_us < start_us:
                # skip until we reach start_us
                t0_us += int(current_dt_ms * 1000)
                continue
            if t0_us > end_us:
                # beyond valid window -> stop
                break

        dt_us = int(current_dt_ms * 1000)
        t1_us = t0_us + dt_us

        # make sure buffer covers up to t1_us
        while len(ts_us_buf) == 0 or ts_us_buf[-1] < t1_us:
            if not load_more():
                break
            trim_buffer()

        if len(ts_us_buf) == 0:
            break

        start_idx = np.searchsorted(ts_us_buf, t0_us, side="left")
        end_idx = np.searchsorted(ts_us_buf, t1_us, side="left")

        if end_idx > start_idx:
            bx = x_buf[start_idx:end_idx]
            by = y_buf[start_idx:end_idx]
            bp = pol_buf[start_idx:end_idx]

            # build event frame
            event_frame = to_event_frame(bx, by, bp, H, W)

            # actual dt for this slice
            dt_ms_here = (ts_us_buf[end_idx - 1] - ts_us_buf[start_idx]) * 1e-3

            # -------------------------------------------------
            # Slice IMU / PX4 data in the same time window
            # -------------------------------------------------
            dv_imu_slice = {}
            if dv_imu_group is not None and dv_imu_ts is not None:
                dv_imu_slice = slice_group_by_time(
                    dv_imu_group, dv_imu_ts, t0_us, t1_us, ts_key_name="timestamp"
                )

            px4_imu_slice = {}
            if px4_imu_group is not None and px4_imu_ts is not None:
                px4_imu_slice = slice_group_by_time(
                    px4_imu_group, px4_imu_ts, t0_us, t1_us, ts_key_name="Timestamp"
                )

            px4_state_slice = {}
            if px4_state_group is not None and px4_state_ts is not None:
                px4_state_slice = slice_group_by_time(
                    px4_state_group, px4_state_ts, t0_us, t1_us, ts_key_name="Timestamp"
                )

            # -------------------------------------------------
            # Yield structured data
            # -------------------------------------------------
            yield {
                "event_frame": event_frame,
                "t0_us": int(t0_us),
                "t1_us": int(t1_us),
                "dt_ms": float(dt_ms_here),
                "dv_imu": dv_imu_slice,
                "px4_imu": px4_imu_slice,
                "px4_state": px4_state_slice,
            }

            # -------------------------------------------------
            # Adaptive slicing: update dt_ms based on this slice
            # -------------------------------------------------
            if slicing_type.lower() == "adaptive" and adaptive_slicer_pid is not None:
                try:
                    n_events = end_idx - start_idx
                    # Assume API: update(dt_ms, n_events) -> new_dt_ms
                    new_dt = adaptive_slicer_pid.update(current_dt_ms, n_events)
                    if new_dt is not None and new_dt > 0:
                        current_dt_ms = float(new_dt)
                except Exception as e:
                    print(f"[DVX][WARN] adaptive_slicer_pid.update failed: {e}")

        # move to next window
        t0_us = t1_us
        trim_buffer()

    f_ev.close()
    print("[DVX] Iterator finished, HDF5 closed.")
