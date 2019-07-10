pushd ../implementations/tensorflow

echo Install requirements
pip install -r requirements.txt

echo Download and build TensorFlow
cc/configure_tensorflow.sh

echo Install TensorFlow package
pip install -U cc/tensorflow_pkg/*.whl

echo Install Horovod
pip install horovod

echo Build minigo cc
./build.sh

echo Install GSUtil
pip2 install gsutil

echo Download checkpoint data and target model
export PYTHONPATH=$(pwd)/ml_perf/tools/tensorflow_quantization/quantization:$PYTHONPATH
BOARD_SIZE=9 python3 ml_perf/get_data.py

popd
