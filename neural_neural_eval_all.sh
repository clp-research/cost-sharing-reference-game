#!/bin/bash

split="test"
#split="val"
map="12"
map2="21"
map3="27"
#model="pretrained"
model="additive-rnd"
args_list=(
  "$split $map $model 49184 0"
  "$split $map $model 98506 0"
  "$split $map $model 92999 0"
  "$split $map2 $model 49184 0"
  "$split $map2 $model 98506 0"
  "$split $map2 $model 92999 0"
  "$split $map3 $model 49184 0"
  "$split $map3 $model 98506 0"
  "$split $map3 $model 92999 0"
)
echo
echo "==================================================="
echo "Starting with args:"
for args in "${args_list[@]}"; do
  echo "$args"
done
echo "==================================================="
echo
for args in "${args_list[@]}"; do
  screen_name=$(echo "$args" | tr ' ' '-')
  if screen -ls | grep -q "$screen_name" >/dev/null; then
    screen -S "$screen_name" -X kill
    echo "Restart screen $screen_name"
  else
    echo "Starting screen ${screen_name}"
  fi
  screen -dmS "${screen_name}" bash -c "./neural_neural_eval.sh ${args}"
done
echo "============================================================================"
echo "All sessions started. See screen -ls. After run screens close automatically."
echo "============================================================================"
