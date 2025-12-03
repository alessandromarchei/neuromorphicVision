#!/usr/bin/env python3
"""
Convert a folder with:

    events.txt
        # timestamp(s) x y polarity
        # 1545313677.546009613 109 241 1
        # 1545313677.546081613 104 237 1

    imu.txt (from davis event based camera)
        # index timestamp(s) ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z
        # 0 1545313677.543659687042 -0.0607... -0.0010... -0.0213... 0.6610... -6.8545... -5.8438...
        #
        # or possibly (no index):
        # timestamp(s) ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z

    groundtruth.txt
        # #timestamp[us] px py pz qx qy qz qw
        # 2.342751200000000000e+07 -4.9012... 4.3717... -9.5029... -8.2272... 4.2370... -1.7456... 3.3633...

    t_offset_us.txt
        # single value, e.g. 1.545313677528878000e+15

into a MVSEC-like HDF5 file with groups:

    /cam/events/{x,y,t,p}
    /cam/imu/{timestamp,ax,ay,az,gx,gy,gz}
    /gt/state/{timestamp,px,py,pz,qx,qy,qz,qw}

where:
  - event timestamps t are in microseconds relative to the offset:
        t_us = round(timestamp_sec * 1e6) - t_offset_us
  - IMU timestamps are treated similarly.
  - GT timestamps are already in microseconds and are stored as-is.

Usage:
  python fpv_to_mvsec_hdf5.py --input_dir path/to/scene_folder \
                              --output path/to/scene.hdf5

If --output is not given, it will create <scene_name>.hdf5 in the input_dir.
"""

import argparse
import logging
from pathlib import Path
import multiprocessing as mp

import numpy as np
import h5py

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logger = logging.getLogger("fpv_to_mvsec_hdf5")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fmt = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)

CAM_TO_IMU_SHIFT = 0.00573388930048  # seconds, FROM THE FPV CALIBRATION FILES
# ---------------------------------------------------------------------
# Helpers for progress bars
# ---------------------------------------------------------------------
def get_tqdm(iterable=None, total=None, desc=None, unit="it"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


# ---------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------
def create_extendable_dataset(group, name, dtype, chunk_size=1024):
    """
    Create an extendable 1D dataset under group with unlimited length.
    """
    return group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        chunks=(chunk_size,),
        dtype=dtype,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def append_to_dataset(dset, data):
    """
    Append 1D data to an extendable dataset.
    """
    data = np.asarray(data)
    if data.size == 0:
        return
    old_size = dset.shape[0]
    new_size = old_size + data.shape[0]
    dset.resize((new_size,))
    dset[old_size:new_size] = data


# ---------------------------------------------------------------------
# File discovery and offset reading
# ---------------------------------------------------------------------
def find_required_files(input_dir: Path):
    """
    Ensure that the required FPV files exist in the input_dir.
    """

    if "outdoor" in input_dir.name.lower():
        logger.info("Detected OUTDOOR scene from folder name.")
        logger.info("Only groundtruth_us.txt will be used for GT.")

        
    events_txt = input_dir / "events.txt"
    imu_txt = input_dir / "imu.txt"
    gt_txt = input_dir / "groundtruth.txt"
    offset_txt = input_dir / "t_offset_us.txt"

    missing = []
    for f in [events_txt, imu_txt, gt_txt, offset_txt]:
        if not f.is_file():
            missing.append(str(f))

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(f" - {m}" for m in missing)
        )

    return events_txt, imu_txt, gt_txt, offset_txt


def read_offset_us(offset_path: Path) -> int:
    """
    Read the single microsecond offset from t_offset_us.txt.
    """
    with open(offset_path, "r") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Offset file is empty: {offset_path}")
    offset = float(text)
    offset_int = int(round(offset))
    logger.info(f"Read t_offset_us = {offset_int} from {offset_path.name}")
    return offset_int


# ---------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------
def _events_worker(events_path: Path, t_offset_us: int, chunk_size: int, queue: mp.Queue):
    """
    Worker that parses events.txt and pushes chunks of (t,x,y,p) into a queue.
    """
    try:
        buf_t, buf_x, buf_y, buf_p = [], [], [], []
        with open(events_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 4:
                    # Unexpected line, skip or log
                    continue
                ts_sec = float(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                p = int(parts[3])

                t_us = int(round(ts_sec * 1e6)) - t_offset_us

                buf_t.append(t_us)
                buf_x.append(x)
                buf_y.append(y)
                buf_p.append(p)

                if len(buf_t) >= chunk_size:
                    arr_t = np.asarray(buf_t, dtype=np.int64)
                    arr_x = np.asarray(buf_x, dtype=np.uint16)
                    arr_y = np.asarray(buf_y, dtype=np.uint16)
                    arr_p = np.asarray(buf_p, dtype=np.int8)
                    queue.put((arr_t, arr_x, arr_y, arr_p))
                    buf_t, buf_x, buf_y, buf_p = [], [], [], []

        # flush last chunk
        if buf_t:
            arr_t = np.asarray(buf_t, dtype=np.int64)
            arr_x = np.asarray(buf_x, dtype=np.uint16)
            arr_y = np.asarray(buf_y, dtype=np.uint16)
            arr_p = np.asarray(buf_p, dtype=np.int8)
            queue.put((arr_t, arr_x, arr_y, arr_p))

    except Exception as e:
        logger.error(f"Error in events worker: {e}")
        queue.put(("__ERROR__", str(e)))

    finally:
        # sentinel
        queue.put(None)


def _imu_worker(imu_path: Path, t_offset_us: int, chunk_size: int, queue: mp.Queue):
    """
    Worker that parses imu.txt and pushes chunks of
    (timestamp_us, ax, ay, az, gx, gy, gz) into a queue.
    """
    try:
        buf_t, buf_ax, buf_ay, buf_az, buf_gx, buf_gy, buf_gz = [], [], [], [], [], [], []
        with open(imu_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # Possible formats:
                #  index ts gx gy gz ax ay az  -> len == 8
                #  ts gx gy gz ax ay az       -> len == 7
                if len(parts) == 8:
                    # ignore index
                    ts_sec = float(parts[1])
                    gx = float(parts[2])
                    gy = float(parts[3])
                    gz = float(parts[4])
                    ax = float(parts[5])
                    ay = float(parts[6])
                    az = float(parts[7])
                elif len(parts) == 7:
                    ts_sec = float(parts[0])
                    gx = float(parts[1])
                    gy = float(parts[2])
                    gz = float(parts[3])
                    ax = float(parts[4])
                    ay = float(parts[5])
                    az = float(parts[6])
                else:
                    # unexpected format
                    continue
                

                #calculate timestamp with imu shift (provided by calibration)
                imu_shift_us = int(round(CAM_TO_IMU_SHIFT * 1e6))  # ≈ 5733 us
                t_us = (int(round(ts_sec * 1e6)) - t_offset_us) + imu_shift_us


                buf_t.append(t_us)
                buf_ax.append(ax)
                buf_ay.append(ay)
                buf_az.append(az)
                buf_gx.append(gx)
                buf_gy.append(gy)
                buf_gz.append(gz)

                if len(buf_t) >= chunk_size:
                    queue.put(
                        (
                            np.asarray(buf_t, dtype=np.int64),
                            np.asarray(buf_ax, dtype=np.float32),
                            np.asarray(buf_ay, dtype=np.float32),
                            np.asarray(buf_az, dtype=np.float32),
                            np.asarray(buf_gx, dtype=np.float32),
                            np.asarray(buf_gy, dtype=np.float32),
                            np.asarray(buf_gz, dtype=np.float32),
                        )
                    )
                    buf_t, buf_ax, buf_ay, buf_az, buf_gx, buf_gy, buf_gz = (
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                    )

        if buf_t:
            queue.put(
                (
                    np.asarray(buf_t, dtype=np.int64),
                    np.asarray(buf_ax, dtype=np.float32),
                    np.asarray(buf_ay, dtype=np.float32),
                    np.asarray(buf_az, dtype=np.float32),
                    np.asarray(buf_gx, dtype=np.float32),
                    np.asarray(buf_gy, dtype=np.float32),
                    np.asarray(buf_gz, dtype=np.float32),
                )
            )

    except Exception as e:
        logger.error(f"Error in IMU worker: {e}")
        queue.put(("__ERROR__", str(e)))

    finally:
        queue.put(None)


# ---------------------------------------------------------------------
# Writing functions
# ---------------------------------------------------------------------
def write_cam_events(h5file: h5py.File, events_path: Path, t_offset_us: int,
                     chunk_size: int = 1_000_000):
    """
    Stream events from events.txt into /cam/events/{x,y,t,p} using a
    multiprocessing worker to parse the text and a consumer to write HDF5.
    """
    logger.info(f"Reading events from: {events_path.name}")

    g_cam = h5file.require_group("cam")
    g_events = g_cam.create_group("events")

    ds_x = create_extendable_dataset(g_events, "x", np.uint16, chunk_size=chunk_size)
    ds_y = create_extendable_dataset(g_events, "y", np.uint16, chunk_size=chunk_size)
    ds_t = create_extendable_dataset(g_events, "t", np.int64, chunk_size=chunk_size)
    ds_p = create_extendable_dataset(g_events, "p", np.int8, chunk_size=chunk_size)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=8)
    proc = ctx.Process(
        target=_events_worker,
        args=(events_path, t_offset_us, chunk_size, queue),
        daemon=True,
    )
    proc.start()

    total_events = 0
    bar = get_tqdm(desc="Events", unit="ev", total=None)

    while True:
        item = queue.get()
        if item is None:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__ERROR__":
            # worker error
            raise RuntimeError(f"Events worker error: {item[1]}")

        arr_t, arr_x, arr_y, arr_p = item
        append_to_dataset(ds_t, arr_t)
        append_to_dataset(ds_x, arr_x)
        append_to_dataset(ds_y, arr_y)
        append_to_dataset(ds_p, arr_p)
        total_events += arr_t.shape[0]
        if bar is not None:
            bar.update(arr_t.shape[0])

    if bar is not None:
        bar.close()
    proc.join()
    logger.info(f"Total events saved: {total_events}")


def write_imu(h5file: h5py.File, imu_path: Path, t_offset_us: int,
              chunk_size: int = 100_000):
    """
    Stream IMU from imu.txt into /imu/{timestamp,ax,ay,az,gx,gy,gz} using
    a multiprocessing worker.
    """
    logger.info(f"Reading IMU from: {imu_path.name}")

    #put the imu data inside hte cam group
    g_cam = h5file.require_group("cam")
    g_imu = g_cam.create_group("imu")

    ds_t = create_extendable_dataset(g_imu, "timestamp", np.int64, chunk_size=chunk_size)
    ds_ax = create_extendable_dataset(g_imu, "ax", np.float32, chunk_size=chunk_size)
    ds_ay = create_extendable_dataset(g_imu, "ay", np.float32, chunk_size=chunk_size)
    ds_az = create_extendable_dataset(g_imu, "az", np.float32, chunk_size=chunk_size)
    ds_gx = create_extendable_dataset(g_imu, "gx", np.float32, chunk_size=chunk_size)
    ds_gy = create_extendable_dataset(g_imu, "gy", np.float32, chunk_size=chunk_size)
    ds_gz = create_extendable_dataset(g_imu, "gz", np.float32, chunk_size=chunk_size)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=8)
    proc = ctx.Process(
        target=_imu_worker,
        args=(imu_path, t_offset_us, chunk_size, queue),
        daemon=True,
    )
    proc.start()

    total_imu = 0
    bar = get_tqdm(desc="IMU", unit="meas", total=None)

    while True:
        item = queue.get()
        if item is None:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__ERROR__":
            raise RuntimeError(f"IMU worker error: {item[1]}")

        (
            arr_t,
            arr_ax,
            arr_ay,
            arr_az,
            arr_gx,
            arr_gy,
            arr_gz,
        ) = item

        append_to_dataset(ds_t, arr_t)
        append_to_dataset(ds_ax, arr_ax)
        append_to_dataset(ds_ay, arr_ay)
        append_to_dataset(ds_az, arr_az)
        append_to_dataset(ds_gx, arr_gx)
        append_to_dataset(ds_gy, arr_gy)
        append_to_dataset(ds_gz, arr_gz)

        total_imu += arr_t.shape[0]
        if bar is not None:
            bar.update(arr_t.shape[0])

    if bar is not None:
        bar.close()
    proc.join()
    logger.info(f"Total IMU samples saved: {total_imu}")


def _load_gt_file(gt_path: Path):
    """
    Load GT file of the form:
        timestamp_us px py pz qx qy qz qw
    into numpy arrays.
    """
    logger.info(f"Loading GT from: {gt_path.name}")
    ts_list = []
    px_list = []
    py_list = []
    pz_list = []
    qx_list = []
    qy_list = []
    qz_list = []
    qw_list = []

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            ts_us_f = float(parts[0])
            ts_us = int(round(ts_us_f))
            px = float(parts[1])
            py = float(parts[2])
            pz = float(parts[3])
            qx = float(parts[4])
            qy = float(parts[5])
            qz = float(parts[6])
            qw = float(parts[7])

            ts_list.append(ts_us)
            px_list.append(px)
            py_list.append(py)
            pz_list.append(pz)
            qx_list.append(qx)
            qy_list.append(qy)
            qz_list.append(qz)
            qw_list.append(qw)

    if len(ts_list) == 0:
        logger.warning(f"No GT data found in {gt_path}")
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    t = np.asarray(ts_list, dtype=np.int64)
    px = np.asarray(px_list, dtype=np.float64)
    py = np.asarray(py_list, dtype=np.float64)
    pz = np.asarray(pz_list, dtype=np.float64)
    qx = np.asarray(qx_list, dtype=np.float64)
    qy = np.asarray(qy_list, dtype=np.float64)
    qz = np.asarray(qz_list, dtype=np.float64)
    qw = np.asarray(qw_list, dtype=np.float64)

    # Ensure sorted by timestamp (just in case)
    order = np.argsort(t)
    t = t[order]
    px = px[order]
    py = py[order]
    pz = pz[order]
    qx = qx[order]
    qy = qy[order]
    qz = qz[order]
    qw = qw[order]

    logger.info(f"Loaded {t.shape[0]} GT samples from {gt_path.name}")
    return t, px, py, pz, qx, qy, qz, qw


def write_gt(h5file: h5py.File, gt_path: Path):
    """
    Write ground truth into /gt/state/{timestamp,px,py,pz,qx,qy,qz,qw}
    """
    g_gt = h5file.create_group("gt")
    g_state = g_gt.create_group("state")

    t, px, py, pz, qx, qy, qz, qw = _load_gt_file(gt_path)

    g_state.create_dataset("timestamp", data=t, compression="gzip", compression_opts=4, shuffle=True)
    g_state.create_dataset("px", data=px, compression="gzip", compression_opts=4, shuffle=True)
    g_state.create_dataset("py", data=py, compression="gzip", compression_opts=4, shuffle=True)
    g_state.create_dataset("pz", data=pz, compression="gzip", compression_opts=4, shuffle=True)
    g_state.create_dataset("qx", data=qx, compression="gzip", compression_opts=4, shuffle=True)
    g_state.create_dataset("qy", data=qy, compression="gzip", compression_opts=4, shuffle=True)
    g_state.create_dataset("qz", data=qz, compression="gzip", compression_opts=4, shuffle=True)
    g_state.create_dataset("qw", data=qw, compression="gzip", compression_opts=4, shuffle=True)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert FPV text logs (events.txt, imu.txt, GT) to MVSEC-style HDF5."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing events.txt, imu.txt, stamped_groundtruth_us*.txt, t_offset_us.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 path. If not given, uses <scene_name>.hdf5 in the input folder.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir does not exist: {input_dir}")

    events_txt, imu_txt, gt_txt, offset_txt = find_required_files(input_dir)
    t_offset_us = read_offset_us(offset_txt)

    # Derive scene name from folder if output not given
    if args.output is None:
        scene_name = input_dir.name
        out_path = input_dir / f"{scene_name}.hdf5"
    else:
        out_path = Path(args.output).resolve()

    logger.info("============================================")
    logger.info(f" Input directory : {input_dir}")
    logger.info(f" Events file     : {events_txt.name}")
    logger.info(f" IMU file        : {imu_txt.name}")
    logger.info(f" GT file         : {gt_txt.name}")
    logger.info(f" Offset file     : {offset_txt.name}")
    logger.info(f" Output HDF5     : {out_path}")
    logger.info("============================================")

    # Create HDF5 and fill it
    with h5py.File(out_path, "w") as h5f:
        write_cam_events(h5f, events_txt, t_offset_us)
        write_imu(h5f, imu_txt, t_offset_us)
        write_gt(h5f, gt_txt)

    logger.info("============================================")
    logger.info(" Conversion completed successfully.")
    logger.info("============================================")


if __name__ == "__main__":
    main()
