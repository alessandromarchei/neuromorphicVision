import os
import subprocess

# --- Configuration ---

REMOTE_HOST = "badile13"   # hostname or alias in ~/.ssh/config
REMOTE_PATH = "/scratch/amarchei/fpv"
LOCAL_PATH = "/home/alessandro/Politecnico Di Torino Studenti Dropbox/Alessandro Marchei/paper/lis_nv/datasets/fpv"  # destination folder

# Files to include

FILES_TO_COPY = [
"calib_undist.txt",
"events.txt",
"images_timestamps_us.txt",
"images.txt",
"imu.txt",
"rectify_map.h5",
"t_offset_us.txt",
"groundtruth.txt",
"stamped_groundtruth_us_cam.txt",
"stamped_groundtruth_us.txt"
]

def main():
# Ensure destination exists
    os.makedirs(LOCAL_PATH, exist_ok=True)

    # 1. List all remote directories under fpv
    print(f"Listing remote directories in {REMOTE_HOST}:{REMOTE_PATH} ...")
    cmd_list = [
        "ssh", REMOTE_HOST, f"find {REMOTE_PATH} -maxdepth 1 -type d -name '*_with_gt'"
    ]
    try:
        remote_dirs = subprocess.check_output(cmd_list, text=True).strip().splitlines()
    except subprocess.CalledProcessError as e:
        print("Error while listing directories:", e)
        return

    print(f"Found {len(remote_dirs)} '_with_gt' folders")

    # 2. Iterate and rsync only the desired files
    for folder in remote_dirs:
        folder_name = os.path.basename(folder)
        local_target = os.path.join(LOCAL_PATH, folder_name)
        os.makedirs(local_target, exist_ok=True)

        print(f"\n=== Copying from {folder_name} ===")

        # Build rsync command for selected files
        for f in FILES_TO_COPY:
            remote_file = f"{REMOTE_HOST}:{os.path.join(folder, f)}"
            cmd_rsync = [
                "rsync", "-avz", "--progress", remote_file, local_target
            ]
            subprocess.run(cmd_rsync)

    print("\n✅ All selected data files copied successfully.")


if __name__ == "__main__":
    main()
