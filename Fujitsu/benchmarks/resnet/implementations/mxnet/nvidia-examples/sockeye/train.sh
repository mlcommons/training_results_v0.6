#!/bin/bash

NUM_GPUS=$(nvidia-smi -L | grep "GPU" | wc -l)

python -m sockeye.train \
        --source corpus.tc.BPE.de \
        --target corpus.tc.BPE.en \
        --validation-source newstest2016.tc.BPE.de \
        --validation-target newstest2016.tc.BPE.en \
        --output models \
        --num-words 50000 \
        --num-layers 2 \
        --rnn-num-hidden 512 \
        --num-embed 512 \
        --max-seq-len 100 \
        --batch-size 64 \
        --bucket-width 10 \
        --checkpoint-frequency 1000 \
        --max-num-checkpoint-not-improved 8 \
        --rnn-dropout-states 0.3 \
        --optimizer adam \
        --gradient-clipping-threshold 1.0 \
        --learning-rate-scheduler-type plateau-reduce \
        --learning-rate-reduce-factor 0.5 \
        --learning-rate-reduce-num-not-improved 3 \
        --rnn-attention-type dot \
        --learning-rate-half-life 10 \
        --optimized-metric perplexity \
        --seed 42 \
        --device-ids -${NUM_GPUS} \
        --max-updates 1500000 \
        --overwrite-output

