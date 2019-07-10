MXNet
=====

MXNet is a deep learning framework designed for both efficiency and
flexibility. It allows you to mix the flavors of symbolic programming and
imperative programming to maximize efficiency and productivity. In its core is a
dynamic dependency scheduler that automatically parallelizes both symbolic and
imperative operations on the fly. A graph optimization layer on top of that
makes symbolic execution fast and memory efficient. The library is portable and
lightweight, and it scales to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of
blueprints and guidelines for building deep learning systems and interesting
insights of DL systems for hackers.

## Contents of the MXNet image

This container has the MXNet framework installed and ready to use.
`/opt/mxnet` contains the complete source of this version of MXNet.

Additionally, this container image also includes several MXNet examples,
which you can find in the `/workspace/examples` directory.

## Running MXNet

You can choose to use MXNet as provided by NVIDIA, or you can choose to
customize it.

MXNet is run simply by importing it as a Python module:
```
$ python
Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import mxnet as mx
>>> a = mx.nd.ones((2,3), mx.gpu())
>>> print (a*2).asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

## Customizing MXNet

You can customize MXNet one of two ways:

(1) Modify the version of the source code in this container and run your
customized version, or (2) use `docker build` to add your customizations on top
of this container if you want to add additional packages.

NVIDIA recommends option 2 for ease of migration to later versions of the
MXNet container image.

For more information, see https://docs.docker.com/engine/reference/builder/ for
a syntax reference.  Several example Dockerfiles are provided in the container
image in `/workspace/docker-examples`.

## Suggested Reading

For more information about MXNet, including tutorials, documentation and examples,
see the `/workspace/examples` directory in the current container,
[MXNet tutorials](http://mxnet.io/tutorials/index.html),
[MXNet HowTo](http://mxnet.io/how_to/index.html), and
[MXNet API](http://mxnet.io/api/python/index.html).
