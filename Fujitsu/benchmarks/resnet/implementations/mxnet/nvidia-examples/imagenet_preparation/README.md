# ImageNet dataset preparation

This example shows how to download and prepare ImageNet (or other custom) dataset to use with MXNet.

## Contents

1. [Download ImageNet dataset](#download-imagenet)
2. [Create lst files](#create-lst-files)
3. [Create RecordIO files](#create-recordio-files)
4. [Next steps](#next-steps)

## Download ImageNet dataset

To download the ImageNet dataset you may use the `download_imagenet.sh` script in this directory.

1. Create an ImageNet account at `http://image-net.org`. You will need a user ID and the access key provided upon registration.
2. Run the `download-imagenet.sh` script. You will be asked for your ImageNet user ID, ImageNet password, and the directory in which to store the dataset. Future re-running of this script will be optimized in that if the tarballs containing the dataset are already available in the target directory, they won't be re-downloaded. However, the unzipping of the tarballs will still take place, so if you already ran this script, don't run it again. For the purpose of the rest of this example, we will assume you downloaded ImageNet to `/data/imagenet` directory, which should now contain `train-jpeg` and `val-jpeg` subdirectories.

## Create lst files

To make a RecordIO file that will be used in training, we first need to create `.lst` files for both training and validation images. To do that, we can use MXNet's `im2rec.py` utility:
```bash
python /opt/mxnet/tools/im2rec.py --list True --recursive True train /data/imagenet/train-jpeg
python /opt/mxnet/tools/im2rec.py --list True --recursive True val /data/imagenet/val-jpeg
```

## Create RecordIO files

RecordIO is a input file format used by MXNet to achieve high performance when loading data (see [http://mxnet.io/architecture/note_data_loading.html] for further details).

We will make 2 RecordIO files - with training and validation data, respectively. We will also pre-resize our images, so that the shorter edge has at least 256px size. To achieve that, we will again use MXNet's `im2rec.py` utility:

```bash
python /opt/mxnet/tools/im2rec.py --resize 256 --quality 95 --num-thread 40 train /data/imagenet/train-jpeg
python /opt/mxnet/tools/im2rec.py --resize 256 --quality 95 --num-thread 40 val /data/imagenet/val-jpeg
```

Here we set JPEG quality setting to 95/100 and used 40 threads to accelerate conversion.

# Next steps

- Image classification example

