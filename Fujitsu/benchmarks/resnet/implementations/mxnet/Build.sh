#!/bin/bash
# Copyright FUJITSU LIMITED 2019

#$ -l rt_C.large=1
#$ -l h_rt=0:10:00
#$ -j y
#$ -cwd

if [ $# -lt 1 ] ; then
    echo "usage: $0 prod|debug [numThread]" >&2
    exit 1
fi

export LANG=C

echo started at `date`

./Build.core "$@"

echo finished at `date`

# End of file

