# MxNet-TensorRT Runtime Integration
## What is this?

This directory contains examples of using the [MxNet](http://mxnet.incubator.apache.org/)-[TensorRT](https://developer.nvidia.com/tensorrt) runtime integration to accelerate model inference.

## How do I get started?

Currently, this directory contains two examples: one for LeNet-5 (trained on [MNIST](http://yann.lecun.com/exdb/mnist/)), and another for ResNet-50 (trained on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)). Try them out simply by running the scripts. Both scripts run inference using MxNet's handling of all graph nodes, as well as using MxNet's graph pass to substitute as much as possible of the graph and to run it using TensorRT. The execution time comparison can be seen at the console output.

To run the LeNet5 model, simply run:
```
python test_tensorrt_lenet5.py
```
The above script works with nosetests, so it can be run this way as well:
```
pip install nose
nosetests test_tensorrt_lenet5.py
```

To run the ResNet-50 test, the model needs to be first trained using the examples from the image-classification directory. The following script will download the data, train the model, and then run inference. The default configuration will run for 20 epochs, which should generate a validation accuracy of about 82%. If a pre-trained model already exists, the existing model will be re-used.
```
./train_rn50.sh
```

## Caveats

Expecting speed-ups for tiny models, such as LeNet-5 on MNIST, is unreasonable. Running inference on the MNIST validation set takes very little time (a fraction of a second), and can be overshadowed by the time it takes to optimize and compile the TensorRT graph. Hence, testing a larger model, e.g. ResNet-50 on CIFAR-10, gives a much better perspective of what practical speed-ups can be expected.
