#!/bin/bash
set -e

##############################################
#           CONFIGURATION SECTION
##############################################

BASE_YAML="config/config_fpv_abmof.yaml"
TMP_DIR="tmp_fpv_abmof"
PY_SCRIPT="playback_fpv_evs.py"
OUT_DIR="runs_fpv_abmof"
MAX_PARALLEL=1

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
    # "/home/alessandro/datasets/fpv/outdoor_45_1_davis_with_gt/"
)

DATASET_NAMES=(
    "indoor_45_2"
    "indoor_45_4"
    "indoor_45_9"
    "indoor_45_12"
    "indoor_45_13"
    "indoor_45_14"
    # "outdoor_45_1"
)



# ABMOF PARAMETERS:
#   enable: true
#   area_number: 8   #divide the image in area_number x area_number cells
#   areaEventThr_init: 1000
#   areaEventThr_min: 100
#   areaEventThr_max: 3000
#   step_ratio: 0.05   # FACTOR to increase or decrease areaEventThr. 5% as in the paper
#   H: 260
#   W: 346
#   radius: 12    #radius used for the matching distance and the clipping of OF vectors



# sweeping : radius
RADIUS_LIST=(3)

##############################################
#              FUNCTIONS
##############################################

edit_yaml() {
    local yaml_file="$1"
    local scene_path="$2"
    local radius_val="$3"

    sed -i "s|scene:.*|scene: \"$scene_path\"|" "$yaml_file"
    sed -i "s|type:.*|type: \"adaptive_abmof\"|" "$yaml_file"
    sed -i "s|visualizeImage:.*|visualizeImage: false|" "$yaml_file"


    sed -i "s|^[[:space:]]*radius:.*|  radius: $radius_val|" "$yaml_file"

}

wait_for_slots() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 5
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

    for radius_val in "${RADIUS_LIST[@]}"; do

        ts=$(timestamp)
        run_id="fpv_${dataset_name}_R${radius_val}"
        yaml_job="${TMP_DIR}/${run_id}_${ts}.yaml"

        echo "================================================================="
        echo "RUNNING ADAPTIVE: $run_id"
        echo "Dataset: $dataset"
        echo "Parameters: RADIUS=$radius_val"
        echo "================================================================="

        wait_for_slots

        cp "$BASE_YAML" "$yaml_job"
        edit_yaml "$yaml_job" "$dataset" "$radius_val"

        python3 "$PY_SCRIPT" --yaml "$yaml_job" --run "$run_id" --out_dir "$OUT_DIR" &
    done
done

wait
echo "=========================================="
echo " FPV ADAPTIVE SLICING SWEEP COMPLETED"
echo "=========================================="
