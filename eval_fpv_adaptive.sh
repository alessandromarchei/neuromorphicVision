#!/bin/bash
set -e

##############################################
#           CONFIGURATION SECTION
##############################################

BASE_YAML="config/config_fpv_adaptive.yaml"
TMP_DIR="tmp_fpv_adaptive"
PY_SCRIPT="playback_fpv_evs.py"
OUT_DIR="runs_fpv_adaptive"
MAX_PARALLEL=6

mkdir -p "$TMP_DIR"
mkdir -p "$OUT_DIR"

# List of FPV scenes
DATASETS=(
    "/home/alessandro/datasets/fpv/indoor_45_2_davis_with_gt/"
    "/home/alessandro/datasets/fpv/indoor_45_4_davis_with_gt/"
    "/home/alessandro/datasets/fpv/indoor_45_9_davis_with_gt/"
    "/home/alessandro/datasets/fpv/indoor_45_12_davis_with_gt/"
    "/home/alessandro/datasets/fpv/indoor_45_13_davis_with_gt/"
    "/home/alessandro/datasets/fpv/indoor_45_14_davis_with_gt/"
    "/home/alessandro/datasets/fpv/outdoor_45_1_davis_with_gt/"
)

DATASET_NAMES=(
    "indoor_45_2"
    "indoor_45_4"
    "indoor_45_9"
    "indoor_45_12"
    "indoor_45_13"
    "indoor_45_14"
    "outdoor_45_1"
)

# PI Sweep Lists (same logic of MVSEC)
P_LIST=(0.3)
I_LIST=(0.05)
# OFPixelSetpoint_LIST=(3 5 7 9 10)
# thresholdPIDEvents_LIST=(5 7 10)
OFPixelSetpoint_LIST=(2 4 8 12)
thresholdPIDEvents_LIST=(5 7 10)

##############################################
#              FUNCTIONS
##############################################

edit_yaml() {
    local yaml_file="$1"
    local scene_path="$2"
    local p_val="$3"
    local i_val="$4"
    local sp_val="$5"
    local th_val="$6"

    sed -i "s|scene:.*|scene: \"$scene_path\"|" "$yaml_file"
    sed -i "s|type:.*|type: \"adaptive\"|" "$yaml_file"
    sed -i "s|visualizeImage:.*|visualizeImage: false|" "$yaml_file"

    sed -i "s|^[[:space:]]*P:.*|  P: $p_val|" "$yaml_file"
    sed -i "s|^[[:space:]]*I:.*|  I: $i_val|" "$yaml_file"
    sed -i "s|^[[:space:]]*OFPixelSetpoint:.*|  OFPixelSetpoint: $sp_val|" "$yaml_file"
    sed -i "s|^[[:space:]]*thresholdPIDEvents:.*|  thresholdPIDEvents: $th_val|" "$yaml_file"
    sed -i "s|^[[:space:]]*adaptiveTimingWindowStep:.*|  adaptiveTimingWindowStep: 3|" "$yaml_file"
}

wait_for_slots() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
}

timestamp() {
    date +%Y%m%d_%H%M%S_%N
}

##############################################
#                 MAIN SWEEP
##############################################

for idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$idx]}"
    dataset_name="${DATASET_NAMES[$idx]}"

    for p in "${P_LIST[@]}"; do
        for i in "${I_LIST[@]}"; do
            for sp in "${OFPixelSetpoint_LIST[@]}"; do
                for th in "${thresholdPIDEvents_LIST[@]}"; do

                    ts=$(timestamp)
                    run_id="fpv_${dataset_name}_P${p}_I${i}_sp${sp}_th${th}"
                    yaml_job="${TMP_DIR}/${run_id}_${ts}.yaml"

                    echo "================================================================="
                    echo "RUNNING ADAPTIVE: $run_id"
                    echo "Dataset: $dataset"
                    echo "P: $p, I: $i, SP: $sp, TH: $th"
                    echo "================================================================="

                    wait_for_slots

                    cp "$BASE_YAML" "$yaml_job"
                    edit_yaml "$yaml_job" "$dataset" "$p" "$i" "$sp" "$th"

                    python3 "$PY_SCRIPT" --yaml "$yaml_job" --run "$run_id" --out_dir "$OUT_DIR" &
                done
            done
        done
    done
done

wait
echo "=========================================="
echo " FPV ADAPTIVE SLICING SWEEP COMPLETED"
echo "=========================================="
