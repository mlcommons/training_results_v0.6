#!/bin/bash
BASE_DIR=$1
FLAG_DIR=$2

NUMA_COUNT=`cat /proc/cpuinfo |grep physical\ id|sort -u |wc -l`
VIRT_CORES=`cat /proc/cpuinfo |grep physical\ id|wc -l`
NUMA_CORES=`cat /proc/cpuinfo |grep cpu\ cores|head -n 1|awk '//{print $4}'`
PHY_CORES=$(expr $NUMA_CORES \* $NUMA_COUNT)

NUM_NODES=`ml_perf/hostlist.sh|wc -l`
TRAIN_NODES=$3
PLAY_NODES=$(expr $NUM_NODES - $TRAIN_NODES)
EVAL_NODES=$PLAY_NODES

# Run training loop
BOARD_SIZE=9  python3  ml_perf/reference_implementation.py \
  --base_dir=$BASE_DIR \
  --flagfile=$FLAG_DIR/rl_loop.flags \
  --physical_cores=$PHY_CORES \
  --virtual_cores=$VIRT_CORES \
  --numa_cores=$NUMA_CORES \
  --quantization=$4 \
  `ml_perf/hostlist.sh |head -n $PLAY_NODES |awk '/./{print "--selfplay_node="$0}'` \
  `ml_perf/hostlist.sh |tail -n $TRAIN_NODES|awk '/./{print "--train_node="$0}'` \
  `ml_perf/hostlist.sh |head -n $EVAL_NODES |awk '/./{print "--eval_node="$0}'`

# Once the training loop has finished, run model evaluation to find the
# first trained model that's better than the target
BOARD_SIZE=9  python3  ml_perf/eval_models.py \
  --base_dir=$BASE_DIR \
  --flags_dir=$FLAG_DIR
