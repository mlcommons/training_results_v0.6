#!/bin/bash
BASE_DIR=$1
FLAG_DIR=$2

NUMA_COUNT=`cat /proc/cpuinfo |grep physical\ id|sort -u |wc -l`
VIRT_CORES=`cat /proc/cpuinfo |grep physical\ id|wc -l`
NUMA_CORES=`cat /proc/cpuinfo |grep cpu\ cores|head -n 1|awk '//{print $4}'`
PHY_CORES=$(expr $NUMA_CORES \* $NUMA_COUNT)

# Run training loop
BOARD_SIZE=9  python3  ml_perf/reference_implementation.py \
  --base_dir=$BASE_DIR \
  --flagfile=$FLAG_DIR/rl_loop.flags \
  --physical_cores=$PHY_CORES \
  --virtual_cores=$VIRT_CORES \
  --numa_cores=$NUMA_CORES \
  --quantization=$3 \
  --train_node=localhost

# Once the training loop has finished, run model evaluation to find the
# first trained model that's better than the target
BOARD_SIZE=9  python3  ml_perf/eval_models.py \
  --base_dir=$BASE_DIR \
  --flags_dir=$FLAG_DIR
