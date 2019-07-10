echo :::MLL `date +%s.%N` submission_org: {\"value\": \"Intel_Corp\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_platform: {\"value\": \"32xCLX-8260L_CPUs\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_division: {\"value\": \"closed\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_status: {\"value\": \"onprem\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_benchmark: {\"value\": \"minigo\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_poc_name: {\"value\": \"Guokai Ma, Letian Kang, Christine Cheng, Mingxiao Huang\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_poc_email: {\"value\": \"guokai.ma@intel.com, letian.kang@intel.com, christine.cheng@intel.com, mingxiao.huang@intel.com\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_entry: {\"value\": {\"framework\": \"TensorFlow 1.13.1\", \"power\": \"none\", \"notes\": \"none\", \"interconnect\": \"OPA\", \"os\": \"Oracle Linux Server 7.6\", \"libraries\": \"MKLDNN \(v0.18\), MKL \(v2019.0.3.20190220\), Intel MPI \(2018.1.163\)\", \"compilers\": \"GCC6.3\", \"nodes\": [{\"num_nodes\": 32, \"cpu\": \"Intel\(R\) Xeon\(R\) Platinum 8260L CPU @ 2.40GHz\", \"num_cores\": 48, \"num_vcpus\": \"NA\", \"accelerator\": \"NA\", \"num_accelerators\": 0, \"sys_mem_size\": \"192G\", \"sys_storage_type\": \"SSD\", \"sys_storage_size\": \"800G\", \"cpu_accel_interconnect\": \"100Gb OPA\", \"network_card\": \"100Gb OPA\", \"num_network_cards\": 1, \"notes\": \"NA\"}]}, \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}

mpirun -np 32 -ppn 1 -f $HOSTLIST.txt /usr/local/bin/cacheclr3
mpirun -np 32 -ppn 1 -f $HOSTLIST.txt /usr/local/bin/echo_1_compact_memory
echo :::MLL `date +%s.%N` cache_clear: {\"value\": \"true\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}

pushd ../implementations/tensorflow-resubmit
# $1: number of training nodes (must < total number of nodes)
# $2: turn on quantization or not
./run_mn.sh 6 True
popd
