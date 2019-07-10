#!/bin/bash

## DL params
EXTRA_PARAMS=(
               --batch-size      "120"
               --eval-batch-size "160"
               --warmup          "650"
               --lr              "2.92e-3"
               --wd              "1.6e-4"
               --use-nvjpeg
               --use-roi-decode
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=01:00:00

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
