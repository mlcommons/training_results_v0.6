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

