#!/bin/bash

case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
0)
    exec numactl --physcpubind=0-1,40-41 --membind=0 "${@}"
    ;;
1)
    exec numactl --physcpubind=2-3,42-43 --membind=0 "${@}"
    ;;
2)
    exec numactl --physcpubind=4-5,44-45 --membind=0 "${@}"
    ;;
3)
    exec numactl --physcpubind=6-7,46-47 --membind=0 "${@}"
    ;;
4)
    exec numactl --physcpubind=8-9,48-49 --membind=0 "${@}"
    ;;
5)
    exec numactl --physcpubind=10-11,50-51 --membind=0 "${@}"
    ;;
6)
    exec numactl --physcpubind=12-13,52-53 --membind=0 "${@}"
    ;;
7)
    exec numactl --physcpubind=14-15,54-55 --membind=0 "${@}"
    ;;
8)
    exec numactl --physcpubind=20-21,60-61 --membind=1 "${@}"
    ;;
9)
    exec numactl --physcpubind=22-23,62-63 --membind=1 "${@}"
    ;;
10)
    exec numactl --physcpubind=24-25,64-65 --membind=1 "${@}"
    ;;
11)
    exec numactl --physcpubind=26-27,66-67 --membind=1 "${@}"
    ;;
12)
    exec numactl --physcpubind=28-29,68-69 --membind=1 "${@}"
    ;;
13)
    exec numactl --physcpubind=30-31,70-71 --membind=1 "${@}"
    ;;
14)
    exec numactl --physcpubind=32-33,72-73 --membind=1 "${@}"
    ;;
15)
    exec numactl --physcpubind=34-35,74-75 --membind=1 "${@}"
    ;;
*)
    echo ==============================================================
    echo "ERROR: Unknown local rank ${OMPI_COMM_WORLD_LOCAL_RANK}"
    echo ==============================================================
    exit 1
    ;;
esac

