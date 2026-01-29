#!/bin/bash
set -e

##############################################
#           CONFIGURATION SECTION
##############################################

BASE_YAML="config/config_mvsec_abmof.yaml"
TMP_DIR="tmp_mvsec_abmof"
PY_SCRIPT="playback_mvsec_evs.py"
OUT_DIR="runs_mvsec_abmof"
MAX_PARALLEL=4

mkdir -p "$TMP_DIR"
mkdir -p "$OUT_DIR"

# List of MVSEC scenes
# Dataset list
DATASETS=(
    "/home/alessandro/datasets/mvsec/indoor_flying/indoor_flying1_data/"
    "/home/alessandro/datasets/mvsec/indoor_flying/indoor_flying2_data/"
    "/home/alessandro/datasets/mvsec/indoor_flying/indoor_flying3_data/"
    "/home/alessandro/datasets/mvsec/outdoor_day1/"
)

# Just labels for run IDs
DATASET_NAMES=(
    "indoor1"
    "indoor2"
    "indoor3"
    "outdoor1"
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
RADIUS_LIST=(3 5 7 8 10 12)

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
        run_id="mvsec_${dataset_name}_R${radius_val}"
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
echo " MVSEC ADAPTIVE SLICING SWEEP COMPLETED"
echo "=========================================="
