Requirements:
* gcc 6.3
* python 3.6 (this should be the default 'python' and 'pip' script)
* python 2.7 (for gsutil)
* numactl
* Intel(R) MPI 2018.1.163
* bazel 0.22

Diskspace:
To run minigo and build all the dependencies, make sure you have 60G space in your $HOME directory and 100G space in your $SCRATCH directory

Setup:
source <path-to-intel-mpi>/bin/mpivars.sh
setup.sh

Execute:
One a single node execution, execute 'run_and_time 2>&1 | tee <log_file>'
One a multiple node execution, do the following 3 steps:
1. source <path-to-intel-mpi>/bin/mpivars.sh
2. Set $SCRATCH to a network shared directory.
3. Save node list to a file named 'hostfile.txt' then execute 'export HOSTLIST=hostfile'.  An alternative way is to modify implementations/tensorflow/ml_perf/hostlist.sh and follow comments
4. Execute 'run_and_time.sh 2>&1 | tee <log_file>'

Performance tuning note:
1. Multinode mode is I/O intensive, make sure to use a high speed shared storage.
2. If the number of nodes changed, the first parameter to 'run_mn.sh' needs to be changed accordingly.  Number of nodes used for training (the first parameter) and nodes used for selfplay (other nodes) needs to be balanced.

Notes:
1. Intel MPI is needed even on single node mode, make sure it is installed in the system

Optimizations:
Selfplay:
1. We ran multiple selfplay instance instead of one selfplay instance with large number of parallel game.
2. Better fusion during freeze graph
3. INT8 quantizatoin for selfplay and evaluation.  Note: target model is kept as FP32.  The quantized model needs to beat FP32 target model to graduate.
4. Split golden chunks to smaller chunks to improve data writing performance.

Training:
1. Random uniform data sampling fusion
2. Data prefetching in data pipeline
3. Batch size = 8192, learning rate = [0.32, 0.032, 0.0032], lr boundaries = [12500, 18750]
4. Distributed/multi-intsance training with Horovod

Training flow:
1. In multi-node mode, pipeline training phase and evaluate+selfplay.  Seperate training and selfplay workload among nodes.

