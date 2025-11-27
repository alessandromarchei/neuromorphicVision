import hdf5plugin
import h5py
import numpy as np

from eval_utils import *
from h5_packager import *


def process_events(h5_file, event_file, delta=100000):
    """
    Copia gli eventi da MVSEC *_data.hdf5 dentro il nuovo .h5 in formato DSEC-like.
    """
    print("Processing events...")

    cnt = 0
    t0 = None
    events = event_file["davis"]["left"]["events"]
    while True:
        x = events[cnt:cnt + delta, 0].astype(np.int16)
        y = events[cnt:cnt + delta, 1].astype(np.int16)
        t = events[cnt:cnt + delta, 2].astype(np.float64)
        p = events[cnt:cnt + delta, 3]
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


def process_flow(h5_file, gt_file, event_file, t0, dt=1):
    """
    Crea i gruppi:
        flow/dt=1/...
        flow/dt=4/...
    usando il GT in *_gt.hdf5 (flow_dist + flow_dist_ts) e
    interpolando tra timestamp APS (image_raw_ts) come in IDNet.

    dt=1  → ogni frame APS
    dt=4  → ogni 4 frame APS
    """
    print(f"Processing flow (dt={dt})...")

    # ----------------------------------------------------
    # Carica GT flow da *_gt.hdf5
    # ----------------------------------------------------
    flow_dist = gt_file["davis"]["left"]["flow_dist"][:]          # (N_gt, 2, H, W)
    flow_ts   = gt_file["davis"]["left"]["flow_dist_ts"][:]       # (N_gt,) in secondi

    flow_x = flow_dist[:, 0, :, :]    # (N_gt, H, W)
    flow_y = flow_dist[:, 1, :, :]    # (N_gt, H, W)
    ts     = flow_ts                  # alias, per chiarezza

    ts_min = ts.min()
    ts_max = ts.max()

    # ----------------------------------------------------
    # Prepara gruppo output nel nuovo .h5
    # ----------------------------------------------------
    group = h5_file.file.create_group(f"flow/dt={dt}")
    ts_table = group.create_dataset(
        "timestamps",
        (0, 2),
        dtype=np.float64,
        maxshape=(None, 2),
        chunks=True,
    )

    flow_cnt = 0
    cur_cnt, prev_cnt = 0, 0
    cur_t, prev_t = None, None

    # Timestamp APS (≈45 Hz outdoor, ≈31 Hz indoor)
    image_raw_ts = event_file["davis"]["left"]["image_raw_ts"][:]  # (N_frames,)

    for t_idx in range(image_raw_ts.shape[0]):
        cur_t = float(image_raw_ts[t_idx])

        # Considera solo il range coperto dal GT
        if cur_t < ts_min:
            continue
        elif cur_t > ts_max:
            break

        # ogni 'dt' frame APS generiamo una nuova mappa di flusso
        if cur_cnt - prev_cnt >= dt:

            # Interpola / propaga il GT 20 Hz sull'intervallo [prev_t, cur_t]
            disp_x, disp_y = estimate_corresponding_gt_flow(
                flow_x,
                flow_y,
                ts,
                prev_t,
                cur_t,
            )
            if disp_x is None:
                # fine GT
                return

            disp = np.stack([disp_x, disp_y], axis=2)  # (H, W, 2)

            # Debug: offset temporale rispetto a t0
            print(f"[dt={dt}] t = {cur_t - t0:.6f} (s)")

            # Salva nel nuovo file .h5
            h5_file.package_flow(disp, (prev_t, cur_t), flow_cnt, dt=dt)
            h5_file.append(ts_table, np.array([[prev_t, cur_t]]))

            # aggiorna contatori
            flow_cnt += 1
            prev_t = cur_t
            prev_cnt = cur_cnt

        cur_cnt += 1
        if prev_t is None:
            prev_t = cur_t


if __name__ == "__main__":
    # ======= MODIFICA QUI I PATH IN BASE ALLA TUA SCENA =======
    indoor_name = "outdoor_day1"
    scene = f"/home/alessandro/datasets/mvsec/outdoor_day1/"  # esempio

    #data is inside scene folder, with *_data.hdf5
    data_path = scene + f"{indoor_name}_data.hdf5"
    gt_path   = scene + f"{indoor_name}_gt.hdf5"
    out_path  = scene + f"{indoor_name}.h5"

    # Carica file MVSEC originali
    event_data = h5py.File(data_path, "r")
    gt_file    = h5py.File(gt_path, "r")

    # Inizializza nuovo file .h5 in formato DSEC-like
    ep = H5Packager(out_path)

    # 1) Copia eventi
    t0, tlast = process_events(ep, event_data)

    # 2) Crea flow upsampled / subsampled da GT HDF5
    process_flow(ep, gt_file, event_data, t0, dt=1)   # ≈45 Hz
    process_flow(ep, gt_file, event_data, t0, dt=4)   # ≈11.25 Hz

    # 3) Metadati
    ep.add_metadata(t0, tlast)

    event_data.close()
    gt_file.close()
