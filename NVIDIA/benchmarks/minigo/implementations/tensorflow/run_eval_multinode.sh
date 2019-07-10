#!/bin/bash

#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

cd minigo
BASE_DIR=$(pwd)/results/
BASE_DIR_SUFFIX=${1:-"$(hostname)-multinode"}
BASE_DIR=$BASE_DIR$BASE_DIR_SUFFIX

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING EVAL RUN AT $start_fmt"

# run benchmark
set -x

echo "running target evaluation for minigo multinode"

# run eval
python ml_perf/eval_models.py \
  --base_dir=$BASE_DIR \
  --flags_dir=ml_perf/flags/9/ ; ret_code=$?

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING EVAL RUN AT $end_fmt"

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# report result
result=$(( $end - $start ))
result_name="REINFORCEMENT"

echo "RESULT,$result_name,$result,nvidia,$start_fmt"
