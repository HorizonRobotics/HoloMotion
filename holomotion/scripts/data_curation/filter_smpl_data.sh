#!/bin/bash

# default json lisy
default_jsonl_list=("humanact12" "MotionX" "OMOMO" "ZJU_Mocap" "amass")
jsonl_list=("${default_jsonl_list[@]}")

# extract command line params
while getopts "l:" opt; do
    case $opt in
        l)
            # 用户输入的 jsonl_list
            IFS=' ' read -r -a jsonl_list <<< "$OPTARG"
            ;;
        *)
            echo "Usage: $0 [-l \"file1 file2 ...\"]"
            exit 1
            ;;
    esac
done

echo "Running label_data.py first..."
python ./holomotion/src/data_curation/filter/label_data.py \
    --jsonl_list "${jsonl_list[@]}"

echo "label_data.py finished."
echo "=============================="

for json in "${jsonl_list[@]}"; do
    echo "Processing $json"

    # 
    if [[ "$json" == "amass" ]]; then
        parent_folder="./data/amass_compatible_datasets/amass"
    else
        parent_folder="./data/amass_compatible_datasets"
    fi

    # 生成路径
    json_path="./data/dataset_labels/${json}.jsonl"
    yaml_path="./holomotion/config/data_curation/${json}_excluded.yaml"

    # 调用 python 脚本
    python ./holomotion/src/data_curation/filter/filter.py \
        --parent_folder "$parent_folder" \
        --json_path "$json_path" \
        --yaml_path "$yaml_path"

    echo "Finished $json"
    echo "-----------------------"
done

echo "All done"