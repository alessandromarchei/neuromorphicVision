
import torch
import numpy as np
import glob
import cv2
import os.path as osp
import torch.utils.data as data
import multiprocessing
import time
import h5py
import os
import hdf5plugin # required for import of blosc
import json
import torch.nn.functional as F
from tqdm import tqdm

import torchvision

from utils.event_utils import to_voxel_grid


def load_intrinsics_ecd(path):
    path_undist = osp.join(path, "calib_undist.txt")
    
    intrinsics = np.loadtxt(path_undist)
    assert len(intrinsics) == 4
    return intrinsics



def get_ecd_data(tss_imgs_us, evs, intrinsics, rectify_map, DELTA_MS=None, H=180, W=240, return_dict=None):
    print(f"Delta {DELTA_MS} ms")
    data_list = []
    print("Creating voxel grids")
    print(f"Number of slices : {len(tss_imgs_us)}")
    for (ts_idx, ts_us) in enumerate(tss_imgs_us):
        if ts_idx == len(tss_imgs_us) - 1:
            break
        
        if DELTA_MS is None:
            t0_us, t1_us = ts_us, tss_imgs_us[ts_idx+1]
        else:
            t0_us, t1_us = ts_us, ts_us + DELTA_MS*1e3
        evs_idx = np.where((evs[:, 0] >= t0_us) & (evs[:, 0] < t1_us))[0]
             
        if len(evs_idx) == 0:
            print(f"no events in range {ts_us*1e-3} - {tss_imgs_us[ts_idx+1]*1e-3} milisecs")
            continue
        evs_batch = np.array(evs[evs_idx, :]).copy()

        if rectify_map is not None:
            rect = rectify_map[evs_batch[:, 2].astype(np.int32), evs_batch[:, 1].astype(np.int32)]
            voxel = to_voxel_grid(rect[..., 0], rect[..., 1], evs_batch[:, 0], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
        else:
            voxel = to_voxel_grid(evs_batch[:, 1], evs_batch[:, 2], evs_batch[:, 0], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
        # visualize_voxel(voxel)
        # img = render(evs_batch[:, 1], evs_batch[:, 2], evs_batch[:, 3], 180, 240) # 
        data_list.append((voxel, intrinsics, min((t0_us+t1_us)/2, tss_imgs_us[ts_idx+1])))

    if return_dict is not None:
        return_dict.update({tss_imgs_us[0]: data_list})
    else:
        return data_list

def read_rmap(rect_file, H=180, W=240):
    h5file = glob.glob(rect_file)[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    assert rectify_map.shape == (H, W, 2)
    rmap.close()
    return rectify_map



def split_evs_list_by_tss_split(evs, tss_imgs_us_split):
    cors = len(tss_imgs_us_split)
    evs_splits = []
    for i in range(cors-1):
        mask_evs_batch = (evs[:, 0] >= tss_imgs_us_split[i][0]) & (evs[:, 0] < tss_imgs_us_split[i+1][0])
        evs_splits.append(evs[mask_evs_batch, :])
    mask_evs_batch = (evs[:, 0] >= tss_imgs_us_split[-1][0]) & (evs[:, 0] <= tss_imgs_us_split[-1][-1])
    evs_splits.append(evs[mask_evs_batch, :])
    return evs_splits



def fpv_evs_iterator(scenedir, stride=1, dT_ms=None, H=260, W=346, parallel=False, cors=4, tss_gt_us=None):

    evs_file = glob.glob(osp.join(scenedir, "events.txt"))
    assert len(evs_file) == 1

    print(f"Loading events from .txt file")
    evs = np.asarray(np.loadtxt(evs_file[0], delimiter=" ")) # (N, 4) with [ts_sec, x, y, p]
    evs[:, 0] = evs[:, 0] * 1e6
    print(f"Loading events done")

    t_offset_us = np.loadtxt(os.path.join(scenedir, "t_offset_us.txt")).astype(np.int64)
    evs[:, 0] -= t_offset_us

    rect_file = osp.join(scenedir, "rectify_map.h5")
    rectify_map = read_rmap(rect_file, H=H, W=W)

    intrinsics = load_intrinsics_ecd(scenedir)
    fx, fy, cx, cy = intrinsics 
    intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

    tss_imgs_us = sorted(np.loadtxt(osp.join(scenedir, "images_timestamps_us.txt")))
    imstart = 0
    imstop = -1
    if tss_gt_us is not None: # fix for FPV
        dT_imgs = tss_imgs_us[-1]-tss_imgs_us[0]
        dT_gt = tss_gt_us[-1]-tss_gt_us[0]
        if (dT_imgs - dT_gt) > 5*1e6 and (tss_gt_us[0] - tss_imgs_us[0]) > 5e6:
            imstart = np.where(tss_imgs_us > tss_gt_us[0])[0][0]
            imstop = np.where(tss_imgs_us < tss_gt_us[-1])[0][-1]
            print(f"Start reading Voxel from {imstart}, {imstop}, due to much shorter GT")

    if dT_ms is None:
        dT_ms = np.mean(np.diff(tss_imgs_us)) / 1e3
        print(f"Using mean dt between images as dT: {dT_ms} ms")
    assert dT_ms > 3 and dT_ms < 200

    tss_imgs_us = tss_imgs_us[imstart:imstop:stride]

    if parallel:
        tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
        evs_split = split_evs_list_by_tss_split(evs, tss_imgs_us_split)

        processes = []
        return_dict = multiprocessing.Manager().dict()      
        for i in range(cors):
            p = multiprocessing.Process(target=get_ecd_data, args=(tss_imgs_us_split[i].tolist(), evs_split[i], intrinsics, rectify_map, dT_ms, H, W, return_dict))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

        keys = np.array(return_dict.keys())
        order = np.argsort(keys)
        data_list = []
        for k in keys[order]:
            data_list.extend(return_dict[k])
    else:
        data_list = get_ecd_data(tss_imgs_us, evs, intrinsics, rectify_map, dT_ms, H, W)

    print(f"Preloaded {len(data_list)} FPV-UZH voxels, imstart={imstart}, imstop={imstop}, stride={stride}, dT_ms={dT_ms} on {scenedir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cpu(), intrinsics.cpu(), ts_us
