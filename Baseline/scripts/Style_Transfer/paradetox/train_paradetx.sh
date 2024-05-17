current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./Baseline/scripts/Style_Transfer
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./Baseline/scripts
three_levels_up_path="$(dirname "$second_levels_up_path")"  # 回退三级目录: ./Baseline
four_levels_up_path="$(dirname "$three_levels_up_path")"  # 回退三级目录: ./
# 进入工作目录: ./llama_factory
working_path="${four_levels_up_path}"

cd "$working_path"

echo "current_dir: $working_path"

export CUDA_VISIBLE_DEVICES=1

python ./Baseline/models/paradetox/gpt2-paradetox.py