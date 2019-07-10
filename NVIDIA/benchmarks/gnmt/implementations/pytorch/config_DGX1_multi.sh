#!/bin/bash

## System run parms
DGXNNODES=48
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=08:00:00

## DL params
LR=${LR:-"4.5e-3"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-8}
WARMUP_STEPS=${WARMUP_STEPS:-200}
REMAIN_STEPS=${REMAIN_STEPS:-1924}
DECAY_INTERVAL=${DECAY_INTERVAL:-241}
TARGET=${TARGET:-24.0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
NUMEPOCHS=${NUMEPOCHS:-12}
MATH=${MATH:-amp_fp16}
EXTRA_OPTS=${EXTRA_OPTS-"\
   --fused-attention \
   --fused-xentropy \
   --prealloc-mode=once \
   --no-log-all-ranks \
   --enable-apex-allreduce-overlap \
   --apex-message-size 32134259 \
   --apex-num-allreduce-streams 2 \
   "}
export CUDNN_PREFER_SM_COUNT_DBG=64

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0'
DGXNSOCKET=2
BIND_LAUNCH=1
