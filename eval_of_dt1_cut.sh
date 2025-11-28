#!/bin/bash
set -e

##############################################
#           CONFIGURATION SECTION
##############################################

BASE_YAML="config/config_mvsec_dt1.yaml"
TMP_DIR="tmp_configs"
PY_SCRIPT="playback_mvsec_evs.py"
MAX_PARALLEL=3
SLEEP_BETWEEN_EDIT=0   # now unnecessary

mkdir -p "$TMP_DIR"

# Dataset list
DATASETS=(
    # "/home/alessandro/datasets/mvsec/indoor_flying/indoor_flying1_data/"
    # "/home/alessandro/datasets/mvsec/indoor_flying/indoor_flying2_data/"
    # "/home/alessandro/datasets/mvsec/indoor_flying/indoor_flying3_data/"
    "/home/alessandro/datasets/mvsec/outdoor_day1/"
)

DATASET_NAMES=(
    # "indoor1"
    # "indoor2"
    # "indoor3"
    "outdoor1"
)

# Sweep parameters
FEATURES_LIST=(100 150 200 250 300)
MAG_LIST=(10 20)

##############################################
#              FUNCTIONS
##############################################

edit_yaml() {
    local yaml_file="$1"
    local scene_path="$2"
    local desired_features="$3"
    local mag_thresh="$4"

    sed -i "s|scene:.*|scene: \"$scene_path\"|" "$yaml_file"
    sed -i "s|desiredFeatures:.*|desiredFeatures: $desired_features|" "$yaml_file"
    sed -i "s|magnitudeThresholdPixel:.*|magnitudeThresholdPixel: $mag_thresh|" "$yaml_file"
    sed -i "s|visualizeImage:.*|visualizeImage: false|" "$yaml_file"
    sed -i "s|type:.*|type: mvsec|" "$yaml_file"
    sed -i "s|gt_mode:.*|gt_mode: dt1|" "$yaml_file"
    sed -i "s|use_valid_frame_range:.*|use_valid_frame_range: true|" "$yaml_file"  #cut the landing and takeoff parts

}

wait_for_slots() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
}

##############################################
#                 MAIN SWEEP
##############################################

for idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$idx]}"
    dataset_name="${DATASET_NAMES[$idx]}"

    for feat in "${FEATURES_LIST[@]}"; do
        for mag in "${MAG_LIST[@]}"; do

            run_id="mvsec_${dataset_name}_feat${feat}_mag${mag}"

            echo "================================================================="
            echo "RUNNING: $run_id"
            echo "Dataset: $dataset"
            echo "Features: $feat"
            echo "MagnitudeThresh: $mag"
            echo "================================================================="

            # Wait for a free parallel slot
            wait_for_slots

            # UNIQUE YAML COPY FOR THIS JOB
            yaml_job="${TMP_DIR}/${run_id}.yaml"
            cp "$BASE_YAML" "$yaml_job"

            # Modify only this dedicated YAML file
            edit_yaml "$yaml_job" "$dataset" "$feat" "$mag"

            # Launch job with dedicated YAML
            python3 "$PY_SCRIPT" --yaml "$yaml_job" --run "$run_id" --out_dir "runs_mvsec_dt1_cut" &

        done
    done
done

wait

echo "=========================================="
echo "   ALL RUNS COMPLETED SUCCESSFULLY"
echo "=========================================="
