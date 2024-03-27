#!/bin/bash

split="test"
#split="val"
reactivity="1"
#reactivity="4"
args_list=(
  "$split 12 $reactivity 49184 0"
  "$split 12 $reactivity 92999 1"
  "$split 12 $reactivity 98506 2"
  "$split 21 $reactivity 49184 3"
  "$split 21 $reactivity 92999 4"
  "$split 21 $reactivity 98506 5"
  "$split 27 $reactivity 49184 6"
  "$split 27 $reactivity 92999 7"
  "$split 27 $reactivity 98506 0"
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
  screen -dmS "${screen_name}" bash -c "./follower_heuristic_eval.sh ${args}"
done
echo "============================================================================"
echo "All sessions started. See screen -ls. After run screens close automatically."
echo "============================================================================"
