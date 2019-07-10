#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst_dir="${script_dir}/tensorflow"
tmp_dir="${script_dir}/minigo_tf"
tmp_pkg_dir="${script_dir}/tensorflow_pkg"

rm -rfd ${tmp_pkg_dir}

rm -rf ${dst_dir}/*
mkdir -p ${dst_dir}

if [ -d "${script_dir}/../ml_perf/tools" ]; then
  echo "Intel AI tools exist."
else
  git clone https://github.com/IntelAI/tools.git ${script_dir}/../ml_perf/tools/
fi

# TODO(tommadams): we should probably switch to Clang at some point.

if [ -d "${tmp_dir}" ]; then
  pushd "${tmp_dir}"
else
  echo "Cloning tensorflow to ${tmp_dir}"
  git clone https://github.com/tensorflow/tensorflow "${tmp_dir}"

  pushd "${tmp_dir}"

  cherry_pick_tag="02c111ab4269ab73a506164e4b54ba996d28a8cf"
  prev_tag="8be9158c7a701d933bbe532f5d54df17f47a4284"

  git diff "${prev_tag}" "${cherry_pick_tag}" > sample.patch

  commit_tag="961bb02b882a8bb921e5be1c09c34b51fffd25dc"
  echo "Checking out ${commit_tag}"
  git checkout "${commit_tag}"
  git apply sample.patch
  cp ${script_dir}/../ml_perf/tools/tensorflow_quantization/graph_transforms/fuse_quantized_convolution.cc ${tmp_dir}/tensorflow/tools/graph_transforms/
fi

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"
cc_opt_flags="${CC_OPT_FLAGS:--march=native}"

PYTHON_BIN_PATH=`which python`

CC_OPT_FLAGS="${cc_opt_flags}" \
PYTHON_BIN_PATH=${PYTHON_BIN_PATH} \
USE_DEFAULT_PYTHON_LIB_PATH="${USE_DEFAULT_PYTHON_LIB_PATH:-1}" \
TF_NEED_JEMALLOC=${TF_NEED_JEMALLOC:-0} \
TF_NEED_GCP=${TF_NEED_GCP:-0} \
TF_NEED_HDFS=${TF_NEED_HDFS:-0} \
TF_NEED_S3=${TF_NEED_S3:-0} \
TF_NEED_KAFKA=${TF_NEED_KAFKA:-0} \
TF_NEED_CUDA=${TF_NEED_CUDA:-0} \
TF_NEED_GDR=${TF_NEED_GDR:-0} \
TF_NEED_VERBS=${TF_NEED_VERBS:-0} \
TF_NEED_OPENCL_SYCL=${TF_NEED_OPENCL_SYCL:-0} \
TF_NEED_ROCM=${TF_NEED_ROCM:-0} \
TF_CUDA_CLANG=${TF_CUDA_CLANG:-0} \
TF_DOWNLOAD_CLANG=${TF_DOWNLOAD_CLANG:-0} \
TF_NEED_TENSORRT=${TF_NEED_TENSORRT:-0} \
TF_NEED_MPI=${TF_NEED_MPI:-0} \
TF_SET_ANDROID_WORKSPACE=${TF_SET_ANDROID_WORKSPACE:-0} \
TF_NCCL_VERSION=${TF_NCCL_VERSION:-1.3} \
TF_ENABLE_XLA=${TF_ENABLE_XLA:-0} \
./configure

. ${script_dir}/../set_avx2_build
BAZEL_OPTS="-c opt --config=mkl \
            --action_env=PATH \
            --action_env=LD_LIBRARY_PATH \
            $BAZEL_BUILD_OPTS \
            --copt=-DINTEL_MKLDNN"
echo "Building tensorflow package"
bazel build -s $BAZEL_OPTS //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${tmp_pkg_dir}

echo "Tensorflow built-ish"
echo "Unpacking tensorflow package..."
unzip -q ${tmp_pkg_dir}/tensorflow-*.whl -d ${tmp_dir}


echo "Copying tensor flow headers to ${dst_dir}"
cp -r ${tmp_dir}/tensorflow-*.data/purelib/tensorflow/include/* "${dst_dir}"
echo "Building tensorflow libraries"

bazel build -s $BAZEL_OPTS \
    //tensorflow:libtensorflow_cc.so \
    //tensorflow:libtensorflow_framework.so

echo "Copying tensorflow libraries to ${dst_dir}"
cp bazel-bin/tensorflow/libtensorflow_*.so "${dst_dir}"
cp bazel-bin/tensorflow/libtensorflow_*.so.1 "${dst_dir}"

cp `find ${tmp_dir} |grep libiomp5.so` ${dst_dir}
cp `find ${tmp_dir} |grep libmklml_intel.so` ${dst_dir}

popd
