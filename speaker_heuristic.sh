#!/bin/bash

N_ARGS=6

if [ ! $# -eq $N_ARGS ]; then
  echo "Please provide exactly $N_ARGS args:"
  echo "run.sh <arg_map> <arg_data> <arg_follower> <arg_upper> <arg_seed> <arg_gpu>"
  exit 1
fi

arg_map="$1"
arg_data="$2"
arg_follower="$3"
arg_upper="$4"
arg_seed="$5"
arg_gpu="$6"
arg_steps="10"

source prepare_path.sh
python3 neumad/agent_learn.py "$arg_map" "$arg_data" "full" "speaker" \
  -S "neural" \
  -SN "additive-rnd" \
  -F "heuristic" \
  -FN "${arg_follower}_c=$arg_upper" \
  -R "$arg_seed" \
  -T "$arg_steps" \
  -G "$arg_gpu" >>run.log 2>&1
