#!/bin/bash

EPOCH=20
MODEL_PREFIX="resnet50"
SYMBOL="${MODEL_PREFIX}-symbol.json"
PARAMS="${MODEL_PREFIX}-$(printf "%04d" $EPOCH).params"
DATA_DIR="./data"

if [[ ! -f $SYMBOL || ! -f $PARAMS ]]; then
  echo -e "\nTrained model does not exist. Training - please wait...\n"
  python $MXNET_HOME/example/image-classification/train_cifar10.py \
     --network resnet --num-layers 50 --num-epochs ${EPOCH} \
     --model-prefix ./${MODEL_PREFIX}
else
   echo "Pre-trained model exists. Skipping training."
fi

echo "Running inference script."

python test_tensorrt_resnet50.py $MODEL_PREFIX $EPOCH $DATA_DIR
