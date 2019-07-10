# 1. Problem

This task benchmarks reinforcement learning for the 9x9 version of the boardgame go.
The model plays games against itself and uses these games to improve play.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorFlow 19.05-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)

# 2. Directions
## Steps to download and verify data

All training data is generated during the selfplay phase of the RL loop.

The only data to be downloaded are the starting checkpoint and the target model. These are downloaded automatically
before the training starts.

## Steps to launch training

### NVIDIA DGX-1 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

```
docker build --pull -t mlperf-nvidia:minigo .
LOGDIR=<path/to/output/dir> CONT=mlperf-nvidia:minigo DGXSYSTEM=DGX1 ./run.sub
```

### NVIDIA DGX-1 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
multi node submission are in the `config_DGX1_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-1:

1. Build the docker container and push to a docker registry
```
docker build --pull -t <docker/registry>/mlperf-nvidia:minigo.
docker push <docker/registry>/mlperf-nvidia:minigo
```

2. Launch the training
```
source config_DGX1_multi.sh && CONT="<docker/registry>/mlperf-nvidia:minigo" LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1_multi sbatch -N $DGXNNODES -t $WALLTIME --ntasks-per-node $DGXNGPU run.sub
```

# 3. Model
### Publication/Attribution

This benchmark is based on a fork of the minigo project (https://github.com/tensorflow/minigo); which is inspired by the work done by Deepmind with ["Mastering the Game of Go with Deep Neural Networks and
Tree Search"](https://www.nature.com/articles/nature16961), ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270), and ["Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning
Algorithm"](https://arxiv.org/abs/1712.01815). Note that minigo is an
independent effort from AlphaGo, and that this fork is minigo is independent from minigo itself. 


### Reinforcement Setup

This benchmark includes both the environment and training for 9x9 go. There are three primary phases performed in each iteration:

 - Selfplay: the *current best* model plays games against itself to produce board positions for training.
 - Training: train the neural networks selfplay data from several recent models. 
 - Model Evaluation: the *current best* and the most recently trained model play a series of games to establish if the current model should replace the current best
 
 Target evaluation is performed after completing the training (please see the Quality section below for more details).

### Structure

This task has a non-trivial network structure, including a search tree.
A good overview of the structure can be found here: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0. 

### Weight and bias initialization and Loss Function
Network weights are initialized with a fixed checkpoint downloaded before the training starts. Loss function is described here;
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961)

### Optimizer
We use a MomentumOptimizer to train the primary network. 

# 4. Quality

### Quality metric
Quality is measured by the number of games won out of 100 against a fixed target model.
The target model is downloaded before automatically before the training starts.

### Quality target
The target is to win at least 50 out of 100 games against the target model.

### Evaluation frequency
Evaluations are performed after completing the training and are not timed.
Checkpoints from every RL loop iteration are evaluated. 
