#!/bin/bash

N_ARGS=5

if [ ! $# -eq $N_ARGS ]; then
  echo "Please provide exactly $N_ARGS args:"
  echo "run.sh <arg_split> <arg_map> <arg_follower> <arg_upper> <arg_seed> <arg_gpu>"
  exit 1
fi

arg_split_name="$1"
arg_target_map_size="$2"
arg_upper="$3"
arg_seed="$4"
arg_gpu="$5"

source prepare_path.sh
python3 neumad/agent_eval.py "$arg_split_name" "$arg_target_map_size" "12" "didact" "full" "speaker" \
  -S "neural" \
  -SN "additive-rnd" \
  -F "heuristic" \
  -FN "cautious_c=$arg_upper" \
  -R "$arg_seed" \
  -G "$arg_gpu" >>run.log 2>&1
