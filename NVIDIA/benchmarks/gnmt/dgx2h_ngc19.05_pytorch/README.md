## Steps to launch training

### NVIDIA DGX-2 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
single node submission are in the `config_DGX2.sh` script.

Steps required to launch single node training on NVIDIA DGX-2:

```
cd ../implementations/pytorch
docker build --pull -t mlperf-nvidia:rnn_translator .
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PULL=0 DGXSYSTEM=DGX2 ./run.sub
```
