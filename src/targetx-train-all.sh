#/bin/#!/usr/bin/env bash

for HORIZON in "1" "2" "3" "4" "5" "6" "7"
do
  for RESOLUTION in "15m" "30m" "1h" "2h" "4h" "8h" "1d"
  do
      export HORIZON=$HORIZON
      export RESOLUTION=$RESOLUTION
      python3 targetx-trainer.py
    done
  done
done
