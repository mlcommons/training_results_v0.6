#!/bin/bash
NUMA_COUNT=`cat /proc/cpuinfo |grep physical\ id|sort -u |wc -l`
VIRT_CORES=`cat /proc/cpuinfo |grep physical\ id|wc -l`
NUMA_CORES=`cat /proc/cpuinfo |grep cpu\ cores|head -n 1|awk '//{print $4}'`
PHY_CORES=$(expr $NUMA_CORES \* $NUMA_COUNT)

echo Physical cores = $PHY_CORES
echo Virtual cores = $VIRT_CORES
echo NUMA cores = $NUMA_CORES

export KMP_HW_SUBSET=2T
echo KMP_HW_SUBSET = $KMP_HW_SUBSET

output_dir=${SCRATCH:-$(pwd)}
echo Output to ${output_dir}

export KMP_BLOCKTIME=1
export KMP_AFFINITY=compact,granularity=fine
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/cc/tensorflow
ulimit -u 760000

export PYTHONPATH=./ml_perf/tools/tensorflow_quantization/quantization:$PYTHONPATH

./run_minigo_mn.sh ${output_dir}/results/$(hostname) ml_perf/flags/9.mn $1 $2
