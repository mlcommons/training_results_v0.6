# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
OS
* CentOS 7.5

Library
* [ProtoBuf v3.5.0.1](https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.0.1)
* [NAsm](http://www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.xz)
* [LibJpegTurbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases/tag/1.5.3)
* CUDA v9.2.148
* NCCL v2.4.2
* cuDNN v7.5.0

# 2. Direction
## Step to build the MXNet framwork
1. Set the libraries and the binaries path into the `setenv`.
2. Launch `Build.core`.

## Step to download dataset

Download the dataset manually following the instructions from the [ImageNet website](http://image-net.org/download). We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

## Step to launch training
1. The working directory is `mxnet/JobScripts`.
2. Modify the trianing parameters in `parameters`.
3. Launch `Submit` with the number of nodes and the maximum of execution time. This script assignes four processes for each node by default. Hence, modify prosess assignment in `BatchBase` accordiong to your enviroinment. If you launch with 2048 GPUs, then your command is as following:
```
   ./Submit 512 0:05:00
```
# 3. Notes

Basically this measurement was done using the rule of the closed division, except for the following six hyper parameters tuned for a large mini-batch training,

- Three Data Augmentation parameters
    - max-random-area
    - min-random-aspect-ratio
    - max-random-aspect-ratio
- Momentum value of SGD
- Epoch number to change the Data Augmentation parameter set #1 to set #2
- LARS eta parameter

Hyper parameters are defined in `mxnet/JobScripts/parameters` file.
See the parameter file for more details.
