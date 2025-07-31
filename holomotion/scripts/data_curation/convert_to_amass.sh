#!/bin/bash

# 默认原始数据路径
DATA_ROOT="./data/raw_datasets"

# 如果传入参数就覆盖默认
if [ ! -z "$1" ]; then
    DATA_ROOT="$1"
fi

python holomotion/src/data_curation/data_smplify.py --data_root "$DATA_ROOT"