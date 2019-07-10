#!/bin/bash

## DL params
MAX_TOKENS=1280
# LEARNING_RATE="22.5e-4"
# WARMUP_UPDATES=200
LEARNING_RATE="1.732e-3"
WARMUP_UPDATES=550
# EXTRA_PARAMS="--enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 21080883 --parallel-backward-allred-cuda-nstreams 4"
EXTRA_PARAMS="--enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 10540442 --parallel-backward-allred-cuda-nstreams 2 --max-source-positions 80 --max-target-positions 80 --adam-betas (0.86,0.92) "
export PYTORCH_JIT=0

## System run parms
DGXNNODES=60
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=08:00:00

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0'
