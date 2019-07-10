#!/bin/bash

## DL params
EXTRA_PARAMS=(
               --batch-size      "56"
               --eval-batch-size "160"
               --warmup          "650"
               --lr              "3.2e-3"
               --wd              "1.3e-4"
               --num-workers     "3"
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=1:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
