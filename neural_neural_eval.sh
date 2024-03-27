#!/bin/bash

N_ARGS=5

if [ ! $# -eq $N_ARGS ]; then
  echo "Please provide exactly $N_ARGS args:"
  echo "run.sh <arg_split> <arg_map> <arg_model> <arg_seed> <arg_gpu>"
  exit 1
fi

arg_split="$1"
arg_map="$2"
arg_model="$3"
arg_seed="$4"
arg_gpu="$5"

source prepare_path.sh
python3 neumad/agent_eval_ma.py "$arg_split" "$arg_map" "$arg_model" \
  -R "$arg_seed" \
  -G "$arg_gpu" >>run.log 2>&1
