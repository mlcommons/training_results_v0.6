#!/bin/bash
#SBATCH -p mlperf		# partition
#SBATCH -N 1       		# number of nodes
#SBATCH -t 12:00:00		# wall time
#SBATCH -J reinforcement	# job name
#SBATCH --exclusive   		# exclusive node access
#SBATCH --mem=0   		# all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --threads-per-core=2	# HT is on
#SBATCH --cores-per-socket=20	# 20 cores on each socket

DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}

## Data, container and volumes
BENCHMARK=${BENCHMARK:-"reinforcement"}
BENCHMARK_NAME="MINIGO"
CONT=${CONT:-"mlperf-nvidia:$BENCHMARK"}
DATADIR=${DATADIR:-"/raid/data"}
LOGDIR=${LOGDIR:-"/raid/results/$BENCHMARK"}
NEXP=${NEXP:-10} # Default number of times to run the benchmark
SEED=${SEED:-$(od -A n -t d -N 3 /dev/urandom)} # Allows passing SEED in, which is helpful if NEXP=1 ; for NEXP>1 we need to pick different seeds for subsequent runs
SYSLOGGING=1
SYS_LOG_GET="'import mlperf_compliance; from minigo.ml_perf.mlperf_log_utils import mlperf_submission_log; mlperf_submission_log(mlperf_compliance.constants.$BENCHMARK_NAME)'"

## shared dir for data exchange
if [[ $SLURM_NNODES -gt 1 ]]; then
    SHARED_DIR_EXCHANGE="$LOGDIR/scratch/multinode_${SLURM_JOB_ID}"
    mkdir -m 777 -p $SHARED_DIR_EXCHANGE
fi

## DO NOT CHANGE ANYTHING BELOW -- DL params are in run_and_time.sh and config_<system>.sh files

## Load system-specific parameters for benchmark
DGXSYSTEM=${DGXSYSTEM:-"DGX1"}
if [[ ! -f "config_${DGXSYSTEM}.sh" ]]; then
  echo "Unknown system, assuming DGX1"
  DGXSYSTEM="DGX1"
fi
source config_${DGXSYSTEM}.sh

IBDEVICES=${IBDEVICES:-$DGXIBDEVICES}

## Check whether we are running in a slurm env
INSLURM=1
if [[ -z "$SLURM_JOB_ID" ]]; then
  INSLURM=0
  export SLURM_JOB_ID="${DATESTAMP}"
  export SLURM_NNODES=1
  export OMPI_COMM_WORLD_LOCAL_RANK=0
else
  env | grep SLURM
fi
if [[ -z "$SLURM_NTASKS_PER_NODE" ]]; then
  export SLURM_NTASKS_PER_NODE="${DGXNGPU}"
fi
if [[ -z "$SLURM_JOB_ID" || $SLURM_NNODES -eq 1 ]]; then
  # don't need IB if not multi-node
  export IBDEVICES=""
fi

# Create results directory
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -p $(dirname "${LOGFILE_BASE}")

## Docker params
CONTVOLS="-v $DATADIR:/data -v $LOGDIR:/results"
NV_GPU="${NVIDIA_VISIBLE_DEVICES:-$(seq 0 $((${SLURM_NTASKS_PER_NODE:-${DGXNGPU}}-1)) | tr '\n' ',' | sed 's/,$//')}"

DOCKEREXEC="env NV_GPU=${NV_GPU} nvidia-docker run --init --rm --net=host --uts=host --ipc=host --ulimit stack=67108864 --ulimit memlock=-1 --name=cont_${SLURM_JOB_ID} --security-opt seccomp=unconfined $IBDEVICES"


export MLPERF_HOST_OS="$(cat /etc/issue | head -1 | cut -f1-3 -d" ") / $(cat /etc/dgx-release | grep -E "DGX_PRETTY_NAME|DGX_OTA_VERSION" |cut -f2 -d= |cut -f2 -d '"' |paste -sd' ')"


if [[ $SLURM_NNODES -gt 1 ]]; then
    if [[ -z "$SHARED_DIR_EXCHANGE" || ! -d "$SHARED_DIR_EXCHANGE" ]]; then
        echo "Error: invalid scratch directory '$SHARED_DIR_EXCHANGE'"
        exit 1
    fi
    CONTVOLS+=" -v $SHARED_DIR_EXCHANGE:/opt/reinforcement/minigo/piazza"
fi

# FIXME: having the container running as non-root isn't yet tested
CONTAINER_UID=0
CONTAINER_GID=0
MULTINODE_DOCKERRUN_ARGS=(
                 --rm
                 --init
                 --net=host
                 --uts=host
                 --ipc=host
                 --ulimit stack=67108864
                 --ulimit memlock=-1
                 --cap-add=SYS_ADMIN
                 --security-opt seccomp=unconfined
                 -u $CONTAINER_UID:$CONTAINER_GID
                 $IBDEVICES
               )


## Prep run and launch
MASTER_IP=`getent hosts \`hostname\` | cut -d ' ' -f1`
PORT=$((4242 + RANDOM%1000))
SSH=''
SRUN=''
if [[ $INSLURM -eq 1 ]]; then
  hosts=( `scontrol show hostname |tr "\n" " "` )
  SSH='ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $hostn'
  SRUN='srun --mem=0 -N 1 -n 1 -w $hostn'
else
  hosts=( `hostname` )
fi

if [[ $SLURM_NNODES -gt 1 ]]; then
  hosts=( `scontrol show hostname |tr "\n" " "` )
  SSH='ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $hostn'
  SRUN="srun --mem=0 -n $SLURM_NNODES --ntasks-per-node=1 --overcommit"
  SRUNl="$SRUN -l"
fi


##########################################
## Configure multinode
##########################################
if [[ $SLURM_NNODES -gt 1 ]]; then

  # systemd needs to be told not to wipe /dev/shm when the user context ends.
  #$SRUNl grep -iR RemoveIPC /etc/systemd

  # 1. Prepare run dir on all nodes
  mkdir -p /dev/shm/mpi/${SLURM_JOB_ID}; chmod 700 /dev/shm/mpi/${SLURM_JOB_ID}

  # 2. Create mpi hostlist
  rm -f /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
  for hostn in ${hosts[@]}; do
    ##-->for minigo 1-rank per node
    echo "$hostn slots=1" >> /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
  done

  # 3. Create mpi config file
  cat > /dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf <<EOF
plm_rsh_agent = /usr/bin/ssh
plm_rsh_args = -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR -oBatchMode=yes -l ${USER}
orte_default_hostfile = /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
btl_openib_warn_default_gid_prefix = 0
mpi_warn_on_fork = 0
btl_openib_connect_udcm_max_retry = 5000
EOF

  # 4. Distribute config file, host list to all nodes
  tar zcPf /dev/shm/mpi/${SLURM_JOB_ID}.tgz /dev/shm/mpi/${SLURM_JOB_ID}/
  cat /dev/shm/mpi/${SLURM_JOB_ID}.tgz | $SRUNl tar zxPf -
  rm -f /dev/shm/mpi/${SLURM_JOB_ID}.tgz

  # 5. Grab SSH keys and configs on each node
  $SRUNl cp -pr ~/.ssh /dev/shm/mpi/${SLURM_JOB_ID}/

fi

MULTINODE_VARS=(
       -e "CONT=${CONT}"
       -e "DGXSYSTEM=${DGXSYSTEM}"
       -e "SLURM_NNODES=${SLURM_NNODES}"
       -e "SLURM_NTASKS_PER_NODE=1"
       -e "OMPI_MCA_mca_base_param_files=/dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf"
     )


if [[ $SLURM_NNODES -gt 1 ]]; then
    # Pull image
	echo "Creating containers:"
	[[ "${PULL}" != "0" ]] && $SRUNl docker pull $CONT
	$SRUNl nvidia-docker run -d "${MULTINODE_DOCKERRUN_ARGS[@]}" --name=cont_${SLURM_JOB_ID} $CONTVOLS "${MULTINODE_VARS[@]}" $CONT bash -c 'sleep infinity' ; rv=$?
    $SRUNl docker exec cont_${SLURM_JOB_ID} rm -f /etc/shinit

	[[ $rv -ne 0 ]] && echo "ERR: Container launch failed." && exit $rv
	
	sleep 30

else
    # Pull image
    [[ "${PULL}" != "0" ]] && docker pull $CONT

	# Test the base container launch
	pids=();
	for hostn in ${hosts[@]}; do
	  timeout -k 600s 600s \
	    $(eval echo $SRUN) $DOCKEREXEC $CONT python -c 'import tensorflow' &
	  pids+=($!);
	done
	wait "${pids[@]}"
	success=$? ; if [ $success -ne 0 ]; then echo "ERR: Base container launch failed."; exit $success ; fi

    # Launch containers
    pids=(); rets=()
    for hostn in ${hosts[@]}; do
      $(eval echo $SSH) $DOCKEREXEC $CONTVOLS $CONT sleep infinity &
      pids+=($!); rets+=($?);
    done
    success=0; for s in ${rets[@]}; do ((success+=s)); done ; if [ $success -ne 0 ]; then echo "ERR: Container launch failed."; exit $success ; fi
    sleep 30 # Making sure containers have time to launch

    # Disable compat check from further running
    pids=(); rets=()
    for hostn in ${hosts[@]}; do
      $(eval echo $SSH) docker exec cont_${SLURM_JOB_ID} rm -f /etc/shinit &
      pids+=($!);
    done
    wait "${pids[@]}"

fi

if [[ $SLURM_NNODES -gt 1 ]]; then
  # For multinode, each container needs to be able to have access to the host system user's SSH keys & config
  # FIXME: this probably doesn't work unless the UID/GID matches (untested) and/or the container is running as root (default)
  $SRUNl docker exec cont_${SLURM_JOB_ID} bash -c "cp -pr /dev/shm/mpi/${SLURM_JOB_ID}/.ssh ~/ ; chown -R $CONTAINER_UID:$CONTAINER_GID ~/.ssh/ ; chmod 700 ~/.ssh/"
fi

export SEED
export NEXP
for nrun in `seq 1 $NEXP`; do
  (
    echo "Beginning trial $nrun of $NEXP"

    export VARS=(
	    "-e" "SLURM_NNODES=$SLURM_NNODES"
	    "-e" "MLPERF_HOST_OS"
    )

    if [[ $SYSLOGGING -eq 1 ]]; then
	    VARS_STR="${VARS[@]}"
        bash -c "echo -n 'Gathering sys log on ' && hostname && docker exec $VARS_STR cont_${SLURM_JOB_ID} python -c ${SYS_LOG_GET}"
        if [[ $? -ne 0 ]]; then
            echo "ERR: Sys log gathering failed."
            exit 1
        fi
    fi

    ## Clear RAM cache dentries and inodes
    echo "Clearing caches"
    LOG_COMPLIANCE="'import mlperf_compliance; mlperf_compliance.mlperf_log.mlperf_print(mlperf_compliance.constants.CACHE_CLEAR, value=True, stack_offset=0)'"
	pids=(); rets=()
	for hostn in ${hosts[@]}; do
      if [[ $INSLURM -eq 1 ]]; then
        $(eval echo $SSH) bash -c 'sync && sudo /sbin/sysctl vm.drop_caches=3' && \
        $(eval echo $SSH) docker exec cont_${SLURM_JOB_ID} python -c $LOG_COMPLIANCE &
      else
        docker run --init --rm --privileged --entrypoint bash $CONT -c "sync && echo 3 > /proc/sys/vm/drop_caches && python -c $LOG_COMPLIANCE || exit 1" &
      fi
	  pids+=($!); rets+=($?);
	done
	wait "${pids[@]}"
	success=0; for s in ${rets[@]}; do ((success+=s)); done ; if [ $success -ne 0 ]; then echo "ERR: Cache clearing failed."; exit $success ; fi

	## Launching benchmark
    ##only on the main node for multinode mpi
	if [[ $SLURM_NNODES -gt 1 ]]; then

        if [[ -z "$SHARED_DIR_EXCHANGE" || ! -d "$SHARED_DIR_EXCHANGE" ]]; then
            echo "Error: invalid scratch directory '$SHARED_DIR_EXCHANGE'"
            exit 1
        else
            # clean up scratch files to ensure fresh start for each experiment
            rm -rf $SHARED_DIR_EXCHANGE/*
        fi

        # launch training and eval
        hostn="${hosts[0]}"
        BASE_DIR_SUFFIX=$hostn"-"$(date +%Y-%m-%d-%H-%M)
        MPIRUN=( mpirun --allow-run-as-root --bind-to none --oversubscribe ${MULTINODE_VARS[@]//-e/-x} --launch-agent "docker exec cont_${SLURM_JOB_ID} orted" )
		docker exec "${MULTINODE_VARS[@]}" cont_${SLURM_JOB_ID} "${MPIRUN[@]}" ./run_and_time_multinode.sh $BASE_DIR_SUFFIX ;
		docker exec "${MULTINODE_VARS[@]}" cont_${SLURM_JOB_ID} bash ./run_eval_multinode.sh $BASE_DIR_SUFFIX ;

    else
	    pids=();
	    export MULTI_NODE=''
	    for h in `seq 0 $((SLURM_NNODES-1))`; do
          hostn="${hosts[$h]}"
	      echo "Launching on node $hostn"
	      if [[ $SLURM_NNODES -gt 1 ]]; then
	        export MULTI_NODE=" --nnodes=$SLURM_NNODES --node_rank=$h --master_addr=$MASTER_IP --master_port=$PORT"
          else
            export MULTI_NODE=" --master_port=$PORT"
	      fi
          export DOCKERENV=(
             "-e" "DGXSYSTEM=$DGXSYSTEM"
             "-e" "SEED=$SEED"
             "-e" "MULTI_NODE=$MULTI_NODE"
             "-e" "SLURM_JOB_ID=$SLURM_JOB_ID"
             "-e" "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
             "-e" "SLURM_NNODES=$SLURM_NNODES"
             "-e" "MLPERF_HOST_OS=$MLPERF_HOST_OS"
          )
          # Execute command
          set -x
          $(eval echo $SRUN) docker exec "${DOCKERENV[@]}" cont_${SLURM_JOB_ID} ./run_and_time.sh &
	      pids+=($!);
          set +x
	    done
    fi
	wait "${pids[@]}"

  ) |& tee ${LOGFILE_BASE}_$nrun.log

  ## SEED update
  export SEED=$(od -A n -t d -N 3 /dev/urandom)
done

if [[ $SLURM_NNODES -gt 1 ]]; then
    rm -rf $SHARED_DIR_EXCHANGE
fi

# Clean up (note: on SLURM we skip this, as the epilogue will take care of it)
if [[ $INSLURM -eq 0 ]]; then
  docker rm -f cont_${SLURM_JOB_ID}
fi
