echo "Cloning subword-nmt project for byte-pair encoding"

git clone https://github.com/rsennrich/subword-nmt.git

export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH

echo "Installing pip dependencies for NMT example"
pip install matplotlib tensorboard

echo "Downloading German training corpus"
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz

echo "Downloading English training corpus"
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz

echo "Unzipping German corpus"
gunzip corpus.tc.de.gz

echo "Unzipping English corpus"
gunzip corpus.tc.en.gz

echo "Downloading development corpora"
curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf - 

echo "Learning byte-pair encoding vocab. Please wait. This may take a few minutes."
python -m learn_joint_bpe_and_vocab --input corpus.tc.de corpus.tc.en \
       -s 30000 \
       -o bpe.codes \
       --write-vocabulary bpe.vocab.de bpe.vocab.en

echo "Applying byte-pair Encoding to German training text. Please wait. This may take a few minutes."
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < corpus.tc.de > corpus.tc.BPE.de

echo "Applying byte-pair Encoding to English training text. Please wait. This may take a few minutes."
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < corpus.tc.en > corpus.tc.BPE.en

echo "Applying byte-pair Encoding to German test text. Please wait. This may take a few minutes."
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < newstest2016.tc.de > newstest2016.tc.BPE.de

echo "Applying byte-pair Encoding to English test text. Please wait. This may take a few minutes."
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < newstest2016.tc.en > newstest2016.tc.BPE.en

echo "OPTIONAL: Reducing RAM requirements by making training set smaller for perf testing"
head -n 200000 corpus.tc.BPE.de > corpus.tc.BPE.de.tmp && mv corpus.tc.BPE.de.tmp corpus.tc.BPE.de
head -n 200000 corpus.tc.BPE.en > corpus.tc.BPE.en.tmp && mv corpus.tc.BPE.en.tmp corpus.tc.BPE.en

echo "Done with preprocessing"
