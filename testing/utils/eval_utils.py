
import os
import torch
from devo.devo import DEVO
from devo.utils import Timer
from pathlib import Path
import datetime
import numpy as np
import yaml
import glob
import copy
import math
import shutil
from scipy.spatial.transform import Rotation as R
from tabulate import tabulate

from devo.plot_utils import plot_trajectory, fig_trajectory
from devo.plot_utils import save_trajectory_tum_format


from tqdm import tqdm

import torchvision
from utils.load_utils import change_intrinsics_resize

@torch.no_grad()
def run_voxel(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, model="DEVO",use_pyramid=True, **kwargs): 
    
    print(f"use_pyramid: {use_pyramid}")

    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=False, viz_flow=viz_flow, model=model, pyramid=use_pyramid, **kwargs)
    
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            # import matplotlib.pyplot as plt
            # plt.switch_backend('Qt5Agg')
            visualize_voxel(voxel.detach().cpu(), save=True, index=i)
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

    for _ in range(12):
        slam.update()

    poses, tstamps, max_nedges = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, max_nedges



@torch.no_grad()
def run_voxel_advanced(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, **kwargs): 
    
    if kwargs.get("devo_debug", False):
        from devo.devo_debug import DEVO
    else:
        from devo.devo import DEVO

    skip_start = kwargs.get("skip_start", 0)
    skip_end = kwargs.get("skip_end", 0)

    print(f"Voxel dir : {voxeldir}")
    skip_start, skip_end = get_skip_ranges(voxeldir, skip_start, skip_end)

    data_list = list(iterator)
    total_len = len(data_list)
    print(f"Total frames in iterator: {total_len}, skipping first {skip_start} and last {skip_end} frames")

    binning_factor = kwargs.get("binning", 1)
    if binning_factor != 1:
        old_H, old_W = H, W
        H, W = H // binning_factor, W // binning_factor
        H = int(math.floor(H/4)*4)
        W = int(math.floor(W/4)*4)
        resize = torchvision.transforms.Resize((H,  W))

    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=False, viz_flow=viz_flow, **kwargs)

    # store per-frame inference times
    frame_times = []

    for i, (voxel, intrinsics, t) in enumerate(tqdm(data_list)):
        if i < skip_start or i >= (total_len - skip_end):
            continue

        if binning_factor != 1:
            voxel = resize(voxel)
            intrinsics = change_intrinsics_resize(intrinsics, H, W, H_orig=old_H, W_orig=old_W)
            intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device="cuda")

        if timing:
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            t_start.record()

        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

        if timing:
            t_end.record()
            torch.cuda.synchronize()
            elapsed = t_start.elapsed_time(t_end) / 1e3  # seconds
            frame_times.append(elapsed)

    for _ in range(12):
        slam.update()

    poses, tstamps, max_nedges = slam.terminate()

    if timing and frame_times:
        total_time = sum(frame_times)
        avg_time = total_time / len(frame_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\n{voxeldir}")
        print(f"Processed {len(frame_times)} frames")
        print(f"Total time: {total_time:.3f} s")
        print(f"Average per-frame time: {avg_time:.4f} s ({avg_fps:.2f} FPS)")

    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, max_nedges





@torch.no_grad()            
def log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                plot=False, save=True, return_figure=False, rpg_eval=False, stride=1, 
                calib1_eds=None, camID_tumvie=None, outdir=None, expname="", max_diff_sec=0.01, save_csv=False, cfg=None, name=None, step=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results

    # unpack data
    traj_GT, tss_GT_us, traj_est, tss_est_us = data
    train_step, net, dataset_name, scene, trial, cfg, args, max_nedges = hyperparam

    ####### SINCE EVALUATION DATA HAS BEEN SHORTENED DUE TO MEMORY ISSUES,
    ####### THE GT DATA should be truncated as the traj_est length
    ####### ONLY FOR TARTANAIR DATASET
    # if 'tartan' in dataset_name.lower():
    #     traj_GT = traj_GT[:len(traj_est)]
    #     tss_GT_us = tss_GT_us[:len(traj_est)]
    

    # create folders
    if train_step is None:
        if isinstance(net, str) and ".pth" in net:
            train_step = os.path.basename(net.split(".")[0])
        else:
            train_step = -1
    scene_name = '_'.join(scene.split('/')[1:]).title() if "/P0" in scene else scene.title()
    if outdir is None:
        outdir = "results"

    if outdir is None:
        outfolder = make_outfolder(outdir, dataset_name, expname, scene_name, trial, train_step, stride, calib1_eds, camID_tumvie)
    else:
        outfolder = make_outfolder_outdir(outdir, dataset_name, expname, scene_name, trial, cfg)

    # save cfg & args to outfolder
    if cfg is not None:
        with open(f"{outfolder}/cfg.yaml", 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    if args is not None:
        if args is not None:
            with open(f"{outfolder}/args.yaml", 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)

    # compute ATE
    ate_score, evoGT, evoEst = ate_real(traj_GT, tss_GT_us, traj_est, tss_est_us)
    all_results.append(ate_score)
    results_dict_scene[scene].append(ate_score)
    
    # following https://github.com/arclab-hku/Event_based_VO-VIO-SLAM/issues/5
    evoGT = make_evo_traj(traj_GT, tss_GT_us)
    evoEst = make_evo_traj(traj_est, tss_est_us)
    gtlentraj = evoGT.get_infos()["path length (m)"]
    evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
    ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
    MPE = ape_trans.stats["mean"] / gtlentraj * 100
    evoATE = ape_trans.stats["rmse"]*100
    assert abs(evoATE-ate_score) < 1e-5
    R_rmse_deg = -1.0

    if save:
        Path(f"{outfolder}").mkdir(exist_ok=True)
        save_trajectory_tum_format((traj_est, tss_est_us), f"{outfolder}/{scene_name}_Trial{trial+1:02d}.txt")

    if rpg_eval:

        fnamegt, fnameest = run_rpg_eval(outfolder, traj_GT, tss_GT_us, traj_est, tss_est_us)
        abs_stats, rel_stats, _ = load_stats_rpg_results(outfolder)

        # abs errs
        ate_rpg = abs_stats["trans"]["rmse"]*100
        print(f"ate_rpg: {ate_rpg:.04f}, ate_real (EVO): {ate_score:.04f}")
        # assert abs(ate_rpg-ate_score)/ate_rpg < 0.1 # 10%
        R_rmse_deg = abs_stats["rot"]["rmse"]
        MTE_m = abs_stats["trans"]["mean"]

        # traj_GT_inter = interpolate_traj_at_tss(traj_GT, tss_GT_us, tss_est_us)
        # ate_inter, _, _ = ate_real(traj_GT_inter, tss_est_us, traj_est, tss_est_us)
        
        res_str = f"\nATE[cm]: {ate_score:.03f} | R_rmse[deg]: {R_rmse_deg:.03f} | MPE[%/m]: {MPE:.03f} \n"
        # res_str += f"MTE[m]: {MTE_m:.03f} | (ATE_int[cm]: {ate_inter:.02f} | ATE_rpg[cm]: {ate_rpg:.02f}) \n"
        
                    #save on results.csv file, in append mode, the following information based on the scene
            #dataset, scene, patches, opt_window,Rem_window,patch_lt, ate, rot_error_x,rot_error_y, rot_error_z
        if cfg is not None and save_csv is not False:
            if name is None:
                name = "out_eval.csv"
            with open(name, "a") as f:
                f.write(
                    f"{'mvsec'},{scene_name},{cfg['PATCHES_PER_FRAME']},{cfg['OPTIMIZATION_WINDOW']},{cfg['REMOVAL_WINDOW']},{cfg['PATCH_LIFETIME']},{max_nedges},{ate_score},{R_rmse_deg},{MPE}\n")




        write_res_table(outfolder, res_str, scene_name, trial)
    else:
        p = f"{outfolder}/"
        p = os.path.abspath(p)
        os.makedirs(p, exist_ok=True)

        
        fnameGT = os.path.join(p, "stamped_groundtruth.txt")
        f = open(fnameGT, "w")
        f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
        for i in range(len(traj_GT)):
            f.write(f"{tss_GT_us[i]/1e6} {traj_GT[i,0]} {traj_GT[i,1]} {traj_GT[i,2]} {traj_GT[i,3]} {traj_GT[i,4]} {traj_GT[i,5]} {traj_GT[i,6]}\n")
        f.close()

        fnameEst = os.path.join(p, "stamped_traj_estimate.txt")
        f = open(fnameEst, "w")
        f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
        for i in range(len(traj_est)):
            f.write(f"{tss_est_us[i]/1e6} {traj_est[i,0]} {traj_est[i,1]} {traj_est[i,2]} {traj_est[i,3]} {traj_est[i,4]} {traj_est[i,5]} {traj_est[i,6]}\n")
        f.close()


        res_str = f"\nATE[cm]: {ate_score:.03f} | MPE[%/m]: {MPE:.03f}"


        write_res_table(outfolder, res_str, scene_name, trial)

    if plot and outdir is None:
        Path(f"{outfolder}/").mkdir(exist_ok=True)
        pdfname = f"{outfolder}/../{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        plot_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
        shutil.copy(pdfname, f"{outfolder}/{scene_name}_Trial{trial+1:02d}_step_{train_step}_stride_{stride}.pdf")
        # [DEBUG]
        #pdfname = f"{outfolder}/GT_{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        #plot_trajectory((traj_GT, tss_GT_us/1e6), (traj_GT, tss_GT_us/1e6), 
        #                f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
        #                pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
    
    elif plot and outdir is not None:
        Path(f"{outfolder}/").mkdir(exist_ok=True)
        pdfname = f"{outfolder}/{scene_name}_Trial{trial+1:02d}_exp_{expname}_cfg_{cfg['PATCHES_PER_FRAME']}_{cfg['REMOVAL_WINDOW']}_{cfg['PATCH_LIFETIME']}.pdf"
        plot_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
        shutil.copy(pdfname, f"{outfolder}/{scene_name}_Trial{trial+1:02d}_cfg_{cfg['PATCHES_PER_FRAME']}_{cfg['REMOVAL_WINDOW']}_{cfg['PATCH_LIFETIME']}.pdf")

    if return_figure:
        fig = fig_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), f"{dataset_name} {scene_name.replace('_', ' ')} {res_str})",
                            return_figure=True, max_diff_sec=max_diff_sec)
        figures[f"{dataset_name}_{scene_name}"] = fig
    
    print(f"Results for {dataset_name} {scene_name} Trial #{trial+1} {res_str}")

    return all_results, results_dict_scene, figures, outfolder



@torch.no_grad()
def write_raw_results(all_results, outfolder):
    # all_results: list of all raw_results
    os.makedirs(os.path.join(f"{outfolder}/../raw_results"), exist_ok=True)
    with open(os.path.join(f"{outfolder}/../raw_results", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
        f.write(','.join([str(x) for x in all_results]))

@torch.no_grad()
def compute_median_results(results, all_results, dataset_name, outfolder=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results
        
    results_dict = dict([(f"{dataset_name}/{k}", np.median(v)) for (k, v) in results.items()])
    results_dict["AUC"] = np.maximum(1 - np.array(all_results), 0).mean()

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)
    results_dict["AVG"] = np.mean(xs) / 100.0 # cm -> m

    if outfolder is not None:
        with open(os.path.join(f"{outfolder}/../results_dict_latex_{datetime.datetime.now().strftime('%m-%d-%I%p.txt')}"), 'w') as f:
            k0 = list(results.keys())[0]
            num_runs = len(results[k0])
            f.write(' & '.join([str(k) for k in results.keys()]))
            f.write('\n')

 
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(v[i]) for v in results.values()]))
                f.write('\n')

            f.write(f"Medians\n")
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(np.median(v)) for v in results.values()]))
                f.write('\n')

            f.write('\n\n')

    return results_dict