# 1. Problem 

This problem uses Attention mechanisms to do language translation.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.05-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)

# 2. Directions

### Steps to download and verify data

Steps to download the dataset, tokenize it, and convert the reference tokenization to the proper pytorch file format:

```
docker build --pull -t mlperf-nvidia:translation .
mkdir -p $DATADIR/raw_data
docker  run --rm -it  -u $(id -u):$(id -g)  -v $DATADIR:/workspace/translation/examples/translation/wmt14_en_de -v $DATADIR/raw_data:/raw_data mlperf-nvidia:translation
bash run_preprocessing.sh
bash run_conversion.sh
```

### Steps to run and time

## Steps to launch training

### NVIDIA DGX-1 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

```
docker build --pull -t mlperf-nvidia:translation .
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PULL=0 DGXSYSTEM=DGX1 ./run.sub
```

### NVIDIA DGX-2 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
single node submission are in the `config_DGX2.sh` script.

Steps required to launch single node training on NVIDIA DGX-2:

```
docker build --pull -t mlperf-nvidia:translation .
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PULL=0 DGXSYSTEM=DGX2 ./run.sub
```

### NVIDIA DGX-1 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
multi node submission are in the `config_DGX1_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-1:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training
```
source config_DGX1_multi.sh && CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1_multi sbatch -N $DGXNNODES -t $WALLTIME --ntasks-per-node $DGXNGPU run.sub

```

### NVIDIA DGX-2 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
multi node submission are in the `config_DGX2_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2:
```
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training
```
source config_DGX2_multi.sh && CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2_multi sbatch -N $DGXNNODES -t $WALLTIME --ntasks-per-node $DGXNGPU run.sub
```

# 3. Dataset/Environment
### Publication/Attribution
We use WMT17 ende training for tranding, and we evaluate using the WMT 2014 English-to-German translation task. See http://statmt.org/wmt17/translation-task.html for more information. 


### Data preprocessing
We combine all the files together and subtokenize the data into a vocabulary.  

### Training and test data separation
We use the train and evaluation sets provided explicitly by the authors.

### Training data order
We split the data into 100 blocks, and we shuffle internally in the blocks. 


# 4. Model
### Publication/Attribution

This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

### Structure 

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Tranformer model is able to easily capture long-distance dependencies.

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.


### Weight and bias initialization

We have two sets of weights to initialize: embeddings and the transformer network. 

The transformer network is initialized using the standard tensorflow variance initalizer. The embedding are initialized using the tensorflow random uniform initializer. 

### Loss function
Cross entropy loss while taking the padding into consideration, padding is not considered part of loss.

### Optimizer
We use the same optimizer as the original authors, which is the Adam Optimizer. We batch for a single P100 GPU of 4096. 

# 5. Quality

### Quality metric
We use the BLEU scores with data from [Attention is All You Need](https://arxiv.org/abs/1706.03762). 


    https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de


### Quality target
We currently run to a BLEU score (uncased) of 25. This was picked as a cut-off point based on time. 


### Evaluation frequency
Evaluation of BLEU score is done after every epoch.


### Evaluation thoroughness
Evaluation uses all of `newstest2014.en`.
