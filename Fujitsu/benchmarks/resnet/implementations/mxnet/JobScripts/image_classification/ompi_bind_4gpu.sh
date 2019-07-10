#!/bin/bash
# Copyright FUJITSU LIMITED 2019

LogFile="$LOGDIR/stdout.txt"
MyDir=`readlink -f "$0" | xargs dirname`

. "$ParameterFile"

NumEpoch=`awk '/^ *--num-epochs/ { print $2; }' $ParameterFile | xargs echo`

Echo=true
test "$OMPI_COMM_WORLD_RANK" -eq 0 && Echo=echo
$Echo "number of epochs: $NumEpoch"

CmdProf=""

if [ ${OMPI_COMM_WORLD_RANK} -eq 0 ]; then
    env > ${LOGDIR}/host_env_0 &
    pip list > ${LOGDIR}/piplist_0 &
    if [ ${UseProf} -eq 1 ]; then
        CmdProf="nvprof -o ${LOGDIR}/nvprof-rank0.prof"
    fi
fi

case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
    0|1) HCA=mlx5_0 ; Node=0 ;;
    2|3) HCA=mlx5_1 ; Node=1 ;;
esac

export OMPI_MCA_btl_openib_if_include="$HCA"
$Echo numactl --cpunodebind="$Node" --membind="$Node" ${CmdProf} python "$MyDir/train_imagenet.py" "${PARAMS[@]}"

numactl --cpunodebind="$Node" --membind="$Node" ${CmdProf} python "$MyDir/train_imagenet.py" "${PARAMS[@]}" || echo "#ERROR at `hostname` `nvidia-smi`"
# End of file

