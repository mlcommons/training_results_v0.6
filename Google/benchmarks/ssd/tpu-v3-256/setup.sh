#!/bin/bash

sudo apt-get -q update
sudo apt-get -q install -y python3-tk unzip
sudo apt-get -q install python3-pip
sudo apt-get -q install libjsoncpp1


python3 -m virtualenv py3 -p /usr/bin/python3.6
source py3/bin/activate

pip3 install sacrebleu==1.2.11 --progress-bar off
pip3 install tf-estimator-nightly==1.14.0.dev2019052101 --progress-bar off
pip3 install tf-nightly==1.14.1.dev20190518 --progress-bar off
pip3 install mlperf_compliance --progress-bar off
pip3 install --upgrade google-api-python-client --progress-bar off
pip3 install --upgrade oauth2client --progress-bar off
pip3 install google-api-python-client google-cloud google-cloud-bigquery --progress-bar off
pip3 install absl-py --progress-bar off
pip3 install --upgrade tensorflow-probability --progress-bar off


sudo apt-get install -y python3.6-dev
pip install Cython==0.28.4 matplotlib==2.2.2 --progress-bar off
pip install pycocotools==2.0.0 --progress-bar off
pip install Pillow==5.2.0 --progress-bar off
pip install opencv-python==3.4.3.18 --progress-bar off
pip install jsonlib-python3 --progress-bar off

# This coco wheel was built from the PR in the pycocotools repo.
# This contains custom C++ code for pycoco eval used by mask and ssd.
WHEEL_NAME=coco-1.1-cp36-cp36m-linux_x86_64.whl
pip install /tmp/$WHEEL_NAME

alias protoc="/usr/local/bin/protoc"
INSTALL_PROTO="yes"
if protoc --version | grep -q -E --regexp="3.6.1$"; then
  INSTALL_PROTO=""
fi

if [ ! -z $INSTALL_PROTO ]; then
  pushd /tmp
  curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
  unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
  # Move protoc to /usr/local/bin/
  sudo mv protoc3/bin/* /usr/local/bin/
  # Move protoc3/include to /usr/local/include/
  if [ -d /usr/local/include/google/protobuf ]; then
    sudo rm -r /usr/local/include/google/protobuf
  fi
  sudo mv protoc3/include/* /usr/local/include/

  # Optional: change owner
  sudo chown $USER /usr/local/bin/protoc
  sudo chown -R $USER /usr/local/include/google
  popd
fi
