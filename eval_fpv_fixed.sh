#!/bin/bash
set -e

##############################################
#           CONFIGURATION SECTION
##############################################

BASE_YAML="config/config_fpv_fixed.yaml"
TMP_DIR="tmp_fpv_fixed"
PY_SCRIPT="playback_fpv_evs.py"
OUT_DIR="runs_fpv_fixed"
MAX_PARALLEL=4

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

# Sweep parameters
DT_LIST=(60)              # fixed slicing times
COMP_FILTER_LIST=(4.0)  # complementary gains (change as needed)

##############################################
#              FUNCTIONS
##############################################

edit_yaml() {
    local yaml_file="$1"
    local scene_path="$2"
    local dt="$3"
    local compK="$4"

    sed -i "s|scene:.*|scene: \"$scene_path\"|" "$yaml_file"
    sed -i "s|type:.*|type: \"fixed\"|" "$yaml_file"
    sed -i "s|dt_ms:.*|dt_ms: $dt|" "$yaml_file"
    sed -i "s|visualizeImage:.*|visualizeImage: false|" "$yaml_file"

    sed -i "s|^[[:space:]]*complementaryK:.*|  complementaryK: $compK|" "$yaml_file"
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

    for dt in "${DT_LIST[@]}"; do
        for comp in "${COMP_FILTER_LIST[@]}"; do

            ts=$(timestamp)
            run_id="fpv_${dataset_name}_dt${dt}_c${comp}"
            yaml_job="${TMP_DIR}/${run_id}_${ts}.yaml"

            echo "================================================================="
            echo "RUNNING FIXED: $run_id"
            echo "Dataset: $dataset"
            echo "dt: $dt ms"
            echo "ComplementaryK: $comp"
            echo "================================================================="

            wait_for_slots

            cp "$BASE_YAML" "$yaml_job"
            edit_yaml "$yaml_job" "$dataset" "$dt" "$comp"

            python3 "$PY_SCRIPT" --yaml "$yaml_job" --run "$run_id" --out_dir "$OUT_DIR" &

        done
    done
done

wait
echo "=========================================="
echo " FPV FIXED SLICING SWEEP COMPLETED"
echo "=========================================="

