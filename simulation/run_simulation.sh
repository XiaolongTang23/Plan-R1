#!/bin/bash
export PROJECT_ROOT=$(dirname "$(realpath "$0")")/..

SIM_TYPE=$1
PLANNER=$2
SPLIT=$3
CKPT_PATH=$4
CKPT_PATH=$(realpath "$CKPT_PATH")

if [ $# -lt 4 ]; then
  echo "Usage: $0 <sim_type> <planner> <split> <ckpt_path>"
  exit 1
fi

# model_path_param
if [[ "$PLANNER" == "planr1_planner" ]]; then
    model_path_param="planner.${PLANNER}.model_path"
elif [[ "$PLANNER" == "planr1_planner_with_refinement" ]]; then
    model_path_param="planner.${PLANNER}.planr1_planner.model_path"
else
    echo "Unsupported planner type: $PLANNER"
    exit 1
fi

# scenario_builder
if [[ "$SPLIT" == *"val14"* ]]; then
    scenario_builder="nuplan"
elif [[ "$SPLIT" == *"test14-random"* || "$SPLIT" == *"test14-hard"* ]]; then
    scenario_builder="nuplan_challenge"
else
    echo "Unsupported split type: $SPLIT"
    exit 1
fi


python "$PROJECT_ROOT/run_simulation.py" \
    +simulation=$SIM_TYPE \
    planner=$PLANNER \
    scenario_builder=$scenario_builder \
    scenario_filter=$SPLIT \
    worker=ray_distributed \
    worker.threads_per_node=128 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    verbose=true \
    experiment_uid=$PLANNER \
    "$model_path_param=$CKPT_PATH"

