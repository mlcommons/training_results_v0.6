# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

export PYTHONPATH=/workspace/translation:${PYTHONPATH}

if [[ $SLURM_NTASKS_PER_NODE -ne 1 ]]; then
    DISTRIBUTED_INIT_METHOD="--distributed-init-method env://"
else
    DISTRIBUTED_INIT_METHOD="--distributed-world-size 1"
fi

# Includes online scoring
python -m bind_launch --nsockets_per_node ${DGXNSOCKET} \
                      --ncores_per_socket ${DGXSOCKETCORES} \
                      --nproc_per_node $SLURM_NTASKS_PER_NODE $MULTI_NODE \
 train.py ${DATASET_DIR} \
  --seed ${SEED} \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm "0.0" \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr "0.0" \
  --warmup-updates ${WARMUP_UPDATES} \
  --lr ${LEARNING_RATE} \
  --min-lr "0.0" \
  --dropout "0.1" \
  --weight-decay "0.0" \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing "0.1" \
  --max-tokens ${MAX_TOKENS} \
  --max-epoch ${NUMEPOCHS} \
  --target-bleu "25.0" \
  --ignore-case \
  --no-save \
  --update-freq 1 \
  --fp16 \
  --fuse-dropout-add \
  --fuse-relu-dropout \
  ${DISTRIBUTED_INIT_METHOD} \
  ${EXTRA_PARAMS} ; ret_code=$?

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
RESULT=$(( ${END} - ${START} ))
RESULT_NAME="transformer"
echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"
