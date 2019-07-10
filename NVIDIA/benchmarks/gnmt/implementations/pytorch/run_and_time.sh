#!/bin/bash

DGXSYSTEM=${DGXSYSTEM:-"DGX1"}
if [[ -f config_${DGXSYSTEM}.sh ]]; then
  source config_${DGXSYSTEM}.sh
else
  source config_DGX1.sh
  echo "Unknown system, assuming DGX1"
fi
SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
SLURM_JOB_ID=${SLURM_JOB_ID:-$RANDOM}
MULTI_NODE=${MULTI_NODE:-''}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
DATASET_DIR='/data'
PREPROC_DATADIR='/preproc_data'
RESULTS_DIR='gnmt_wmt16'

LR=${LR:-"2.0e-3"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
WARMUP_STEPS=${WARMUP_STEPS:-200}
REMAIN_STEPS=${REMAIN_STEPS:-10336}
DECAY_INTERVAL=${DECAY_INTERVAL:-1296}
TARGET=${TARGET:-24.0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
NUMEPOCHS=${NUMEPOCHS:-8}
EXTRA_OPTS=${EXTRA_OPTS:-""}
BIND_LAUNCH=${BIND_LAUNCH:-1}
MATH=${MATH:-fp16}

if [[ $BIND_LAUNCH -eq 1 ]]; then
  LAUNCH_OPT="bind_launch  --nsockets_per_node ${DGXNSOCKET}  --ncores_per_socket ${DGXSOCKETCORES} --nproc_per_node ${SLURM_NTASKS_PER_NODE} ${MULTI_NODE}" 
else
  LAUNCH_OPT="torch.distributed.launch --nproc_per_node ${SLURM_NTASKS_PER_NODE} ${MULTI_NODE}"
fi

echo "running benchmark"

# run training
python -m ${LAUNCH_OPT}  train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --preproc-data-dir ${PREPROC_DATADIR} \
  --target-bleu $TARGET \
  --epochs "${NUMEPOCHS}" \
  --math ${MATH} \
  --max-length-train ${MAX_SEQ_LEN} \
  --print-freq 10 \
  --train-batch-size $TRAIN_BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --optimizer FusedAdam \
  --lr $LR \
  --warmup-steps $WARMUP_STEPS \
  --remain-steps $REMAIN_STEPS \
  --decay-interval $DECAY_INTERVAL \
  $EXTRA_OPTS ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RNN_TRANSLATOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

