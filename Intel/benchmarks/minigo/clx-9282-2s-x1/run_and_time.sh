echo :::MLL `date +%s.%N` submission_org: {\"value\": \"Intel_Corp\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_platform: {\"value\": \"1xCLX-9282_CPU\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_division: {\"value\": \"closed\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_status: {\"value\": \"onprem\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_benchmark: {\"value\": \"minigo\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_poc_name: {\"value\": \"Guokai Ma, Letian Kang, Christine Cheng, Mingxiao Huang\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_poc_email: {\"value\": \"guokai.ma@intel.com, letian.kang@intel.com, christine.cheng@intel.com, mingxiao.huang@intel.com\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}
echo :::MLL `date +%s.%N` submission_entry: {\"value\": {\"framework\": \"TensorFlow 1.13.1\", \"power\": \"none\", \"notes\": \"none\", \"interconnect\": \"Ethernet\", \"os\": \"CentOS Linux release 7.6.1810 \(Core\)\", \"libraries\": \"MKLDNN \(v0.18\), MKL \(v2019.0.3.20190220\), Intel MPI \(2018.1.163\)\", \"compilers\": \"GCC6.3\", \"nodes\": [{\"num_nodes\": 1, \"cpu\": \"Intel\(R\) Xeon\(R\) Platinum 9282 CPU @ 2.60GHz\", \"num_cores\": 112, \"num_vcpus\": \"NA\", \"accelerator\": \"NA\", \"num_accelerators\": 0, \"sys_mem_size\": \"768G\", \"sys_storage_type\": \"SSD\", \"sys_storage_size\": \"512G\", \"cpu_accel_interconnect\": \"1Gb Ethernet\", \"network_card\": \"1Gb Ethernet\", \"num_network_cards\": 1, \"notes\": \"NA\"}]}, \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}

sync
echo 1 > /proc/sys/vm/compact_memory
echo 3 > /proc/sys/vm/drop_caches
echo :::MLL `date +%s.%N` cache_clear: {\"value\": \"true\", \"metadata\": {\"lineno\": 0, \"file\": \"manual\"}}

pushd ../implementations/tensorflow
# True for quantization
./run.sh True
popd
