#!/bin/bash

case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
0)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=0-2,48-50 --membind=0 "${@}"
    ;;
1)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=3-5,51-53 --membind=0 "${@}"
    ;;
2)
    export OMPI_MCA_btl_openib_if_include=mlx5_2
    exec numactl --physcpubind=6-8,54-56 --membind=0 "${@}"
    ;;
3)
    export OMPI_MCA_btl_openib_if_include=mlx5_2
    exec numactl --physcpubind=9-11,57-59 --membind=0 "${@}"
    ;;
4)
    export OMPI_MCA_btl_openib_if_include=mlx5_3
    exec numactl --physcpubind=12-14,60-62 --membind=0 "${@}"
    ;;
5)
    export OMPI_MCA_btl_openib_if_include=mlx5_3
    exec numactl --physcpubind=15-17,63-65 --membind=0 "${@}"
    ;;
6)
    export OMPI_MCA_btl_openib_if_include=mlx5_4
    exec numactl --physcpubind=18-20,66-68 --membind=0 "${@}"
    ;;
7)
    export OMPI_MCA_btl_openib_if_include=mlx5_4
    exec numactl --physcpubind=21-23,69-71 --membind=0 "${@}"
    ;;
8)
    export OMPI_MCA_btl_openib_if_include=mlx5_7
    exec numactl --physcpubind=24-26,72-74 --membind=1 "${@}"
    ;;
9)
    export OMPI_MCA_btl_openib_if_include=mlx5_7
    exec numactl --physcpubind=27-29,75-77 --membind=1 "${@}"
    ;;
10)
    export OMPI_MCA_btl_openib_if_include=mlx5_8
    exec numactl --physcpubind=30-32,78-80 --membind=1 "${@}"
    ;;
11)
    export OMPI_MCA_btl_openib_if_include=mlx5_8
    exec numactl --physcpubind=33-35,81-83 --membind=1 "${@}"
    ;;
12)
    export OMPI_MCA_btl_openib_if_include=mlx5_9
    exec numactl --physcpubind=36-38,84-86 --membind=1 "${@}"
    ;;
13)
    export OMPI_MCA_btl_openib_if_include=mlx5_9
    exec numactl --physcpubind=39-41,87-89 --membind=1 "${@}"
    ;;
14)
    export OMPI_MCA_btl_openib_if_include=mlx5_10
    exec numactl --physcpubind=42-44,90-92 --membind=1 "${@}"
    ;;
15)
    export OMPI_MCA_btl_openib_if_include=mlx5_10
    exec numactl --physcpubind=45-47,93-95 --membind=1 "${@}"
    ;;
*)
    echo ==============================================================
    echo "ERROR: Unknown local rank ${OMPI_COMM_WORLD_LOCAL_RANK}"
    echo ==============================================================
    exit 1
    ;;
esac

