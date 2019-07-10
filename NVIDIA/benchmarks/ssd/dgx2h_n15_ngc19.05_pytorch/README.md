## Steps to launch training

### NVIDIA DGX-2 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
multi node submission are in the `config_DGX2_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2:

1. Build the docker container and push to a docker registry
```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector
```

2. Launch the training
```
source config_DGX2_multi.sh && CONT="<docker/registry>/mlperf-nvidia:single_stage_detector" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2_multi sbatch -N $DGXNNODES -t $WALLTIME --ntasks-per-node $DGXNGPU run.sub
```
