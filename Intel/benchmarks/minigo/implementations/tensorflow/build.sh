#!/bin/bash
. ./set_avx2_build
bazel build  --incompatible_remove_native_http_archive=false -c opt  --verbose_failures --define=tf=1  --define=board_size=9 $BAZEL_BUILD_OPTS cc:selfplay cc:eval

