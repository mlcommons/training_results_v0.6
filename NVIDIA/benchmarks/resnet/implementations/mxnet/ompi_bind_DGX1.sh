#!/bin/bash

case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
0)
    export OMPI_MCA_btl_openib_if_include=mlx5_0
    exec numactl --physcpubind=0-4,40-44 --membind=0 "${@}"
    ;;
1)
    export OMPI_MCA_btl_openib_if_include=mlx5_0
    exec numactl --physcpubind=5-9,45-49 --membind=0 "${@}"
    ;;
2)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=10-14,50-54 --membind=0 "${@}"
    ;;
3)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=15-19,55-59 --membind=0 "${@}"
    ;;
4)
    export OMPI_MCA_btl_openib_if_include=mlx5_2
    exec numactl --physcpubind=20-24,60-64 --membind=1 "${@}"
    ;;
5)
    export OMPI_MCA_btl_openib_if_include=mlx5_2
    exec numactl --physcpubind=25-29,65-69 --membind=1 "${@}"
    ;;
6)
    export OMPI_MCA_btl_openib_if_include=mlx5_3
    exec numactl --physcpubind=30-34,70-74 --membind=1 "${@}"
    ;;
7)
    export OMPI_MCA_btl_openib_if_include=mlx5_3
    exec numactl --physcpubind=35-39,75-79 --membind=1 "${@}"
    ;;
*)
    echo ==============================================================
    echo "ERROR: Unknown local rank ${OMPI_COMM_WORLD_LOCAL_RANK}"
    echo ==============================================================
    exit 1
    ;;
esac

