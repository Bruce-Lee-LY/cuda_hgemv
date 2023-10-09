# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log ncu && mkdir -p log ncu

# $1: N, $2: K
evaluate_hgemv() {
    echo "Evaluating $1 * $2"
    $WORK_PATH/output/bin/hgemv -N=$1 -K=$2 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/hgemv_${1}_${2}.log 2>&1
    sleep 3
}

# $1: N, $2: K
ncu_hgemv() {
    echo "NCU $1 * $2"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/hgemv_${1}_${2} $WORK_PATH/output/bin/hgemv -N=$1 -K=$2 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_hgemv_${1}_${2}.log 2>&1
    sleep 3
}

benchmark_hgemv() {
    N_dims=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048 3072 4096)
    K=128

    for N in ${N_dims[@]};
    do
        evaluate_hgemv $N $K
        # ncu_hgemv $N $K
    done
}

nohup $WORK_PATH/output/bin/hgemv -N=256 -K=128 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/hgemv_256_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/hgemv_256_128 $WORK_PATH/output/bin/hgemv -N=256 -K=128 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_hgemv_256_128.log 2>&1

# benchmark_hgemv
