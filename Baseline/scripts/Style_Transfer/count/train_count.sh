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



python ./Baseline/models/count/train_seq2seq.py \
        --lm_name /media/data/1/yx/data/model_cache/gpt2-large \
        --model_name gpt2 \
        --save_folder /media/data/3/toxic_model_cache/count-gpt2-large \
        --tensorboard_path /media/data/3/toxic_model_cache/count-gpt2-large/tensorboard \
        --dataset_name  paradetox \
        --contrastive_loss \
        --alpha 0.3 \
        --num_epochs 10