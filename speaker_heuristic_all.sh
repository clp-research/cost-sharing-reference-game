#!/bin/bash

data="didact"
follower="cautious"
args_list=(
  "12 $data $follower 99 49184 0"
  "12 $data $follower 99 98506 1"
  "12 $data $follower 99 92999 2"
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
  screen -dmS "${screen_name}" bash -c "./speaker_heuristic.sh ${args}"
done
echo "============================================================================"
echo "All sessions started. See screen -ls. After run screens close automatically."
echo "============================================================================"
