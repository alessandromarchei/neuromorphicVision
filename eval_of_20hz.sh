#!/bin/bash
set -e

##############################################
#           CONFIGURATION SECTION
##############################################

YAML_FILE="config/config_mvsec_20hz.yaml"
PY_SCRIPT="playback_mvsec_evs.py"          # <--- il tuo script python
MAX_PARALLEL=3                      # max jobs in parallelo
SLEEP_BETWEEN_EDIT=5                # sec

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

# Sweep parameters
FEATURES_LIST=(100 150 200 250 300)
MAG_LIST=(10 20)

##############################################
#              FUNCTIONS
##############################################

edit_yaml() {
    local scene_path="$1"
    local desired_features="$2"
    local mag_thresh="$3"

    # Usa sed per aggiornare i parametri
    sed -i "s|scene:.*|scene: \"$scene_path\"|" "$YAML_FILE"
    sed -i "s|desiredFeatures:.*|desiredFeatures: $desired_features|" "$YAML_FILE"
    sed -i "s|magnitudeThresholdPixel:.*|magnitudeThresholdPixel: $mag_thresh|" "$YAML_FILE"

    #set visualizeImage to false to speed up
    sed -i "s|visualizeImage:.*|visualizeImage: false|" "$YAML_FILE"
}

wait_for_slots() {
    # assicura massimo N processi
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

            run_id="mvsec_${dataset_name}_feat${feat}_mag${mag}_20hz"

            echo "================================================================="
            echo "RUNNING: $run_id"
            echo "Dataset: $dataset"
            echo "Features: $feat"
            echo "MagnitudeThresh: $mag"
            echo "================================================================="

            # 1) Free slot if too many parallel jobs
            wait_for_slots

            # 2) Edit YAML
            edit_yaml "$dataset" "$feat" "$mag"

            # 3) Wait small delay to avoid file race conditions
            sleep "$SLEEP_BETWEEN_EDIT"

            # 4) Launch run in background
            python3 "$PY_SCRIPT" --yaml "$YAML_FILE" --run "$run_id" &

        done
    done
done

# Wait for all jobs
wait

echo "=========================================="
echo "   ALL RUNS COMPLETED SUCCESSFULLY"
echo "=========================================="
