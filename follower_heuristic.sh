#!/bin/bash

N_ARGS=5

if [ ! $# -eq $N_ARGS ]; then
  echo "Please provide exactly $N_ARGS args:"
  echo "run.sh <arg_map> <arg_data> <arg_reactivity> <arg_seed> <arg_gpu>"
  exit 1
fi

arg_map="$1"
arg_data="$2"
arg_reactivity="$3"
arg_seed="$4"
arg_gpu="$5"

source prepare_path.sh
python3 neumad/agent_learn.py "$arg_map" "$arg_data" "full" "follower" \
  -F "neural" \
  -FN "additive-rnd" \
  -S "heuristic" \
  -SN "r=$arg_reactivity" \
  -R "$arg_seed" \
  -T "10" \
  -G "$arg_gpu" >>run.log 2>&1
