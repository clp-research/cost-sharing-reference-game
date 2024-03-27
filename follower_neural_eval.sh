#!/bin/bash

N_ARGS=4

if [ ! $# -eq $N_ARGS ]; then
  echo "Please provide exactly $N_ARGS args:"
  echo "run.sh <arg_split> <arg_map> <arg_seed> <arg_gpu>"
  exit 1
fi

arg_split="$1"
arg_map="$2"
arg_seed="$3"
arg_gpu="$4"

source prepare_path.sh
python3 neumad/agent_eval.py "$arg_split" "$arg_map" "12" "didact" "full" "follower" \
  -F "neural" \
  -FN "additive-rnd" \
  -S "neural" \
  -SN "pretrained" \
  -R "$arg_seed" \
  -G "$arg_gpu" >>run.log 2>&1
