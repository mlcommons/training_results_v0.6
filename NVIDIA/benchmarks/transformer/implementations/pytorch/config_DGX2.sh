#!/bin/bash

## DL params
MAX_TOKENS=8192
LEARNING_RATE="1.976e-3"
WARMUP_UPDATES=1000
EXTRA_PARAMS="--max-source-positions 80 --max-target-positions 80 --enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 147566182 --parallel-backward-allred-cuda-nstreams 2 --adam-betas (0.9,0.98) "

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=01:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
