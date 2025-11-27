import hdf5plugin
import h5py
import numpy as np

from eval_utils import *
from h5_packager import *
import argparse
import os


def process_events(h5_file, event_file, delta=100000):
    print("Processing events...")

    cnt = 0
    t0 = None
    events = event_file["davis"]["left"]["events"]
    while True:
        x = events[cnt : cnt + delta, 0].astype(np.int16)
        y = events[cnt : cnt + delta, 1].astype(np.int16)
        t = events[cnt : cnt + delta, 2].astype(np.float64)
        p = events[cnt : cnt + delta, 3]
        p[p < 0] = 0
        p = p.astype(np.bool_)
        if x.shape[0] <= 0:
            break
        else:
            tlast = t[-1]

        if t0 is None:
            t0 = t[0]

        h5_file.package_events(x, y, t, p)
        cnt += delta

    return t0, tlast
        

def process_flow(h5_file, gt_flow, event_file, t0, dt=1):
    print("Processing flow...")

    group = h5_file.file.create_group("flow/dt=" + str(dt))
    ts_table = group.create_dataset("timestamps", (0, 2), dtype=np.float64, maxshape=(None, 2), chunks=True)
    flow_cnt = 0
    cur_cnt, prev_cnt = 0, 0
    cur_t, prev_t = None, None
    flow_x, flow_y, ts = gt_flow["x_flow_dist"], gt_flow["y_flow_dist"], gt_flow["timestamps"]

    image_raw_ts = event_file["davis"]["left"]["image_raw_ts"]
    for t in range(image_raw_ts.shape[0]):
        cur_t = image_raw_ts[t]

        # upsample flow only at the frame timestamps in between gt samples
        if cur_t < gt_flow["timestamps"].min():
            continue
        elif cur_t > gt_flow["timestamps"].max():
            break

        # skip dt frames between each gt sample
        if cur_cnt - prev_cnt >= dt:

            # interpolate flow
            disp_x, disp_y = estimate_corresponding_gt_flow(
                flow_x, 
                flow_y, 
                ts,
                prev_t,
                cur_t,
            )
            if disp_x is None:
                return
            disp = np.stack([disp_x, disp_y], axis=2)
            print(cur_t - t0)
            h5_file.package_flow(disp, (prev_t, cur_t), flow_cnt, dt=dt)
            h5_file.append(ts_table, np.array([[prev_t, cur_t]]))
            
            # update counters
            flow_cnt += 1
            prev_t = cur_t
            prev_cnt = cur_cnt
        
        cur_cnt += 1
        if prev_t is None:
            prev_t = image_raw_ts[t]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format MVSEC npz/hdf5 files for a given scene.")
    parser.add_argument("dataset_root", type=str, help="Root directory of the dataset containing the scene files")
    args = parser.parse_args()

    dataset_root = args.dataset_root

    # Find scene name from files in dataset_root
    files = os.listdir(dataset_root)
    gt_file = next(f for f in files if f.endswith("_gt_flow_dist.npz"))
    event_file = next(f for f in files if f.endswith("_data.hdf5"))
    scene = gt_file.replace("_gt_flow_dist.npz", "")

    gt_path = os.path.join(dataset_root, gt_file)
    event_data_path = os.path.join(dataset_root, event_file)
    output_path = os.path.join(dataset_root, f"{scene}.h5")

    # load data
    gt = np.load(gt_path)
    event_data = h5py.File(event_data_path, "r")

    # initialize h5 file
    ep = H5Packager(output_path)

    # process events
    t0, tlast = process_events(ep, event_data)
    # process flow
    process_flow(ep, gt, event_data, t0, dt=1)
    process_flow(ep, gt, event_data, t0, dt=4)

    ep.add_metadata(t0, tlast)
 