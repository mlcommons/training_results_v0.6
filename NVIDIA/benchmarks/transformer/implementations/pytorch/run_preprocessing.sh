#!/bin/bash

set -e

SEED=$1

cd /workspace/translation

# TODO: Add SEED to process_data.py since this uses a random generator (future PR)
#export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

mkdir -p /workspace/translation/examples/translation/wmt14_en_de
mkdir -p /workspace/translation/examples/translation/wmt14_en_de/utf8

cp /workspace/translation/reference_dictionary.ende.txt /workspace/translation/examples/translation/wmt14_en_de/dict.en.txt
cp /workspace/translation/reference_dictionary.ende.txt /workspace/translation/examples/translation/wmt14_en_de/dict.de.txt

sed -i "1s/^/\'<lua_index_compat>\'\n/" /workspace/translation/examples/translation/wmt14_en_de/dict.en.txt
sed -i "1s/^/\'<lua_index_compat>\'\n/" /workspace/translation/examples/translation/wmt14_en_de/dict.de.txt

# TODO: make code consistent to not look in two places (allows temporary hack above for preprocessing-vs-training)
cp /workspace/translation/reference_dictionary.ende.txt /workspace/translation/examples/translation/wmt14_en_de/utf8/dict.en.txt
cp /workspace/translation/reference_dictionary.ende.txt /workspace/translation/examples/translation/wmt14_en_de/utf8/dict.de.txt

(cd /workspace/translation/examples/translation/wmt14_en_de && wget https://storage.googleapis.com/tf-perf-public/official_transformer/test_data/newstest2014.tgz && tar -xzvf newstest2014.tgz)

python3 preprocess.py --raw_dir /raw_data/ --data_dir /workspace/translation/examples/translation/wmt14_en_de

