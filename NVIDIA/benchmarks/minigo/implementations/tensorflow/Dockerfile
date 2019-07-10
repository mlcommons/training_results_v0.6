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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:19.05-py3
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update && apt-get install -y --no-install-recommends \
        python2.7 \
        infiniband-diags \
        pciutils && \
    rm -rf /var/lib/apt/lists/*

# Install gsutil
ENV CLOUDSDK_PYTHON /usr/bin/python2.7
RUN curl https://sdk.cloud.google.com | bash -s -- --disable-prompts
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Install Python dependencies
WORKDIR /opt/reinforcement

COPY minigo/requirements.txt requirements.txt
RUN pip install --no-cache-dir https://github.com/mlperf/training/archive/6289993e1e9f0f5c4534336df83ff199bd0cdb75.zip#subdirectory=compliance \
 && pip install --no-cache-dir -r requirements.txt

# Copy MiniGo code and build
COPY . .

ENV BOARD_SIZE 9

RUN cp -r /opt/reinforcement/minigo/cc /opt/tensorflow && \
    cp -r /usr/local/lib/tensorflow /opt/tensorflow/cc && \
    cp -r /usr/local/lib/python3.5/dist-packages/tensorflow/include/* /opt/tensorflow/cc/tensorflow && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs && \
    ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    cd /opt/tensorflow && cp -r /usr/local/mpi cc/ && ./nvbuild.sh --python3.5 --configonly && \
    bazel build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --define=gpu=1 --define=trt=1 --define=tf=1 --define=board_size=${BOARD_SIZE} cc:selfplay cc:eval cc:selfplay_mpi cc:eval_mpi && \
    cp -r bazel-bin /opt/reinforcement/minigo
