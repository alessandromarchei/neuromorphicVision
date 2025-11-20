
import torch
import numpy as np
import glob
import cv2
import os.path as osp
import torch.utils.data as data
import multiprocessing
import h5py
import os
import torch.nn.functional as F
from tqdm import tqdm


from testing.utils.event_utils import to_voxel_grid, to_event_frame

def read_rmap(rect_file, H=180, W=240):
    h5file = glob.glob(rect_file)[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    assert rectify_map.shape == (H, W, 2)
    rmap.close()
    return rectify_map



# def mvsec_evs_iterator(scenedir, side="left", dT_ms=None, H=260, W=346):

#     print(f"loading calibs from {os.path.join(scenedir, f'calib_undist_{side}.txt')}")
#     intrinsics = np.loadtxt(os.path.join(scenedir, f"calib_undist_{side}.txt"))
#     fx, fy, cx, cy = intrinsics
#     intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

#     h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
#     assert len(h5in) == 1
#     datain = h5py.File(h5in[0], 'r')

#     num_imgs = datain["davis"][side]["image_raw"].shape[0]
#     tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"tss_imgs_us_{side}.txt")))
#     assert num_imgs == len(tss_imgs_us)

#     rect_file = osp.join(scenedir, f"rectify_map_{side}.h5")
#     rectify_map = read_rmap(rect_file, H=H, W=W)

#     event_idxs = datain["davis"][side]["image_raw_event_inds"] #image_raw_event_inds : mapping for the nearest event to each DAVIS image in time, 
#     all_evs = datain["davis"][side]["events"][:]
#     evidx_left = 0
#     data_list = []
#     for img_i in range(num_imgs):        
#         #take the event id between the current image and the next image
#         evid_nextimg = event_idxs[img_i]

#         #take the events between the current image and the next image
#         evs_batch = all_evs[evidx_left:evid_nextimg][:]

#         #update the event id for the next image
#         evidx_left = evid_nextimg

#         #rectify the events of the batch
#         rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]

#         #create the voxel grid, with 5 channels
#         if not use_event_stack : 
#             voxel = to_voxel_grid(rect[..., 0], rect[..., 1], evs_batch[:, 2], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
#         else : 
#             voxel = to_event_stack(rect[..., 0], rect[..., 1], evs_batch[:, 2], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)

#         data_list.append((voxel, intrinsics, tss_imgs_us[img_i]))


#     datain.close()

#     if timing:
#         t1.record()
#         torch.cuda.synchronize()
#         dt = t0.elapsed_time(t1)/1e3
#         print(f"Preloaded {len(data_list)} MVSEC-voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
#     print(f"Preloaded {len(data_list)} MVSEC-voxels, imstart={0}, imstop={-1}, stride={1}, dT_ms={dT_ms} on {scenedir}")

#     for (voxel, intrinsics, ts_us) in data_list:
#         yield voxel.cuda(), intrinsics.cuda(), ts_us



def mvsec_evs_iterator(scenedir, side="left", dT_ms=None, H=260, W=346,
                       use_event_stack=False, timing=False):

    print(f"[MVSEC] Loading MVSEC events from {scenedir}")

    # ============================
    # LOAD INPUTS
    # ============================
    h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
    assert len(h5in) == 1
    datain = h5py.File(h5in[0], 'r')

    # All raw events
    all_evs = datain["davis"][side]["events"][:]  
    # Format is (x, y, t_us, pol)

    # Image timestamps (used when dT_ms=None)
    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"tss_imgs_us_{side}.txt")))
    num_imgs = len(tss_imgs_us)

    # Rectification map
    rect_file = osp.join(scenedir, f"rectify_map_{side}.h5")
    rectify_map = read_rmap(rect_file, H=H, W=W)

    # Mapping from image index → event index (used when dt=None)
    event_idxs = datain["davis"][side]["image_raw_event_inds"]

    datain.close()

    # =====================================================================================
    # CASE 1: dT_ms = None → USE ORIGINAL MVSEC IMAGE-BASED SLICING 
    # =====================================================================================
    if dT_ms is None:
        print("[MVSEC] Using default MVSEC slicing (one event frame per image timestamp).")

        evidx_left = 0

        for img_i in range(num_imgs):
            evid_next = event_idxs[img_i]

            # Slice events between image i and i+1
            batch = all_evs[evidx_left:evid_next]
            evidx_left = evid_next

            if batch.shape[0] == 0:
                continue

            # Rectification
            rect = rectify_map[
                batch[:, 1].astype(np.int32),
                batch[:, 0].astype(np.int32)
            ]

            event_frame = to_event_frame(rect[..., 0], rect[..., 1],
                                      batch[:, 2], batch[:, 3],
                                      H=H, W=W)
            
            yield event_frame, tss_imgs_us[img_i]

        return

    # =====================================================================================
    # CASE 2: dT_ms SPECIFIED → TIME-BASED SLICING LIKE FPV
    # =====================================================================================
    print(f"[MVSEC] Using temporal slicing: dT_ms={dT_ms}")

    ts = all_evs[:, 2].astype(np.int64)
    dt_us = int(dT_ms * 1000)

    ts_start = ts[0]
    ts_end   = ts[-1]

    # Uniform timestamps every dt_us
    t0_list = np.arange(ts_start, ts_end, dt_us)

    ev_index = 0
    N = all_evs.shape[0]

    for t0 in t0_list:
        t1 = t0 + dt_us

        # Move ev_index to first event >= t0
        while ev_index < N and ts[ev_index] < t0:
            ev_index += 1
        start = ev_index

        # Move forward until events reach t1
        while ev_index < N and ts[ev_index] < t1:
            ev_index += 1
        end = ev_index

        batch = all_evs[start:end]
        if batch.shape[0] == 0:
            continue

        # Rectification
        rect = rectify_map[
            batch[:, 1].astype(np.int32),
            batch[:, 0].astype(np.int32)
        ]

        # Event frame
        event_frame = to_event_frame(rect[..., 0], rect[..., 1],
                                    batch[:, 2], batch[:, 3],
                                    H=H, W=W)

        yield event_frame, t0



def mvsec_evs_loader(scenedir, side="left", stride=1, H=260, W=346):
    intrinsics = np.loadtxt(os.path.join(scenedir, f"calib_undist_{side}.txt"))
    fx, fy, cx, cy = intrinsics
    intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

    h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
    assert len(h5in) == 1
    datain = h5py.File(h5in[0], 'r')

    num_imgs = datain["davis"][side]["image_raw"].shape[0]
    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"tss_imgs_us_{side}.txt")))
    assert num_imgs == len(tss_imgs_us)

    rect_file = osp.join(scenedir, f"rectify_map_{side}.h5")
    rectify_map = read_rmap(rect_file, H=H, W=W)

    event_idxs = datain["davis"][side]["image_raw_event_inds"]
    all_evs = datain["davis"][side]["events"][:]
    evidx_left = 0
    data_list = []
    for img_i in range(num_imgs):       
        evid_nextimg = event_idxs[img_i]
        evs_batch = all_evs[evidx_left:evid_nextimg][:]
        evidx_left = evid_nextimg
        rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]

        voxel = to_voxel_grid(rect[..., 0], rect[..., 1], evs_batch[:, 2], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
        data_list.append((voxel, intrinsics, tss_imgs_us[img_i]))

    datain.close()

    print(f"Preloaded {len(data_list)} MVSEC-voxels, imstart={0}, imstop={-1}, stride={1} on {scenedir}")

    return data_list
