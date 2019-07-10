# Running TPU submissions (Cloud)

## Setup

### Cloud Environment

In order to run with a cloud TPU, you will need to
[configure your environment](https://cloud.google.com/tpu/docs/quickstart). In
particular, submissions assume that you have:

1.  In particular, it assumes that you have a
    [GCS storage bucket](https://cloud.google.com/tpu/docs/storage-buckets)
    which is located in the same region as your cloud TPU. The TPU's service
    account must access in order to both read the input data and write model
    checkpoints.
2.  The user instance must have
    [permissions](https://cloud.google.com/iam/docs/overview) to access cloud
    TPU APIs.
3.  The project must have [quota](https://cloud.google.com/storage/quotas) to
    create cloud TPUs for the submission.

### Local Environment

This README assumes a clean Ubuntu 16.04 instance running in Google Cloud. When
the submission is run it will perform a variety of system configurations; the
most notable are: 1. Construction of a Virtualenv. - The model will run using
this environment, and it is assumed that no other Python packages have been
installed and the `PYTHONPATH` environment variable is not set. All submissions
are run using Python3.

1.  Installation of GCC.

*   This is not necessary to run the models, but it is needed to run the
    [Cloud TPU Profiler](https://cloud.google.com/tpu/docs/cloud-tpu-tools#profile_tab)
    which can be used to extract performance information from the TPU. However,
    feel free to replace the `upgrade_gcc.sh` script packaged in this submission
    with an empty script.

1.  Set Environment Variables

*   Several environment variables must be set in order for the model script to
    run properly. They are:

```
      # Model directory.
      MLP_GCS_MODEL_DIR

      # Model specific dataset paths. Set this to the location in GCS of the preprocessed dataset
      MLP_PATH_GCS_IMAGENET
      MLP_PATH_GCS_TRANSFORMER
      MLP_PATH_GCS_SSD
      MLP_PATH_GCS_MASKRCNN
      MLP_PATH_GCS_NMT
      MLP_PATH_GCS_NCF
```

### Dataset Preparation

All models save 1 require that the user has already prepared the dataset and
placed it in GCS. Instructions on dataset preparation vary from model to model:

*   [LSTM NMT](https://github.com/mlperf/training/tree/master/rnn_translator#steps-to-download-and-verify-data)
    (Does not need to be converted to TFRecords)
*   Transformer: See TRANSFORMER_DATA_SETUP.md
*   [SSD and Mask-RCNN](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet#preparing-the-coco-dataset)
*   [ResNet](https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment)

In addition to training data, the object detection models require pre-trained
weights for their respective ResNet backbones:

*   The SSD backbone is a ResNet34 backbone trained on the ImageNet dataset. To
    use it set

```
          MLP_GCS_RESNET_CHECKPOINT=gs://download.tensorflow.org/models/mlperf/v0.5.0/resnet34_ssd_checkpoint
```

*   Mask-RCNN uses the ResNet50 pre-trained backbone provided in the
    [Cloud TPU RetinaNet](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet)
    repository. No configuration is necessary.

## Run

Once the environment is configured, simply run `bash run_and_time.sh`

# Research submissions

Research submissions were run using Google internal infrastructure. Contact
