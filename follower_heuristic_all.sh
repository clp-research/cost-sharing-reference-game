#!/bin/bash

data="didact"

args_list=(
  "12 $data 1 49184 0"
  "12 $data 1 98506 1"
  "12 $data 1 92999 2"
  "12 $data 4 49184 3"
  "12 $data 4 98506 4"
  "12 $data 4 92999 5"
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
  screen -dmS "${screen_name}" bash -c "./follower_heuristic.sh ${args}"
done
echo "============================================================================"
echo "All sessions started. See screen -ls. After run screens close automatically."
echo "============================================================================"
