export CUDA_VISIBLE_DEVICES=1

# ---------toxic-------------------------------------------------------------------
data_type='toxic'

#python /media/data/2/yx/model_toxic/Baseline/models/goodtriever/run_all.py \
#  --output_folder /media/data/2/yx/model_toxic/data/datastore_for_googtriever/outputs/goodtriever-large \
#  --prompts_path /media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-test-toxic-2k.jsonl \
#  --model_name /media/data/1/yx/data/model_cache/gpt2-large \
#  --batch_size 4 \
#  --knn True \
#  --knn_temp 100 \
#  --lmbda 2.0 \
#  --dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_toxic \
#  --other_dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_nontoxic
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type ppl
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --ppl_type gpt2-xl \
#    --toxicity_type toxicity

#current_dir=$(cd "$(dirname "$0")" && pwd)
#
#first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./Baseline/scripts/
#second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./Baseline
#three_levels_up_path="$(dirname "$second_levels_up_path")" # 回退两级目录 ./
## 进入工作目录: ./llama_factory
#working_path="${three_levels_up_path}"
#
#cd "$working_path"
#
#echo "current_dir: $working_path"
#export PYTHONPATH="$working_path"
#
#python ./Baseline/run_evaluation_sim_and_fl.py \
#       --eval_type fl \
#       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
#       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/goodtriever-gpt2-l-test-${data_type}-2k.jsonl \
#       --batch_size 128
#
#python ./Baseline/run_evaluation_sim_and_fl.py \
#       --eval_type sim \
#       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
#       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/goodtriever-gpt2-l-test-${data_type}-2k.jsonl \
#       --batch_size 128


#----------------------------nontoxic-------------------------------------------------------------------------------------------
data_type='nontoxic'
#python /media/data/2/yx/model_toxic/Baseline/models/goodtriever/run_all.py \
#  --output_folder /media/data/2/yx/model_toxic/data/datastore_for_googtriever/outputs/goodtriever-large \
#  --prompts_path /media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-test-nontoxic-8k.jsonl \
#  --model_name /media/data/1/yx/data/model_cache/gpt2-large \
#  --batch_size 4 \
#  --knn True \
#  --knn_temp 100 \
#  --lmbda 2.0 \
#  --dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_toxic \
#  --other_dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_nontoxic

#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type ppl
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --ppl_type gpt2-xl \
#    --toxicity_type toxicity

current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./Baseline/scripts/
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./Baseline
three_levels_up_path="$(dirname "$second_levels_up_path")" # 回退两级目录 ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}"

cd "$working_path"

echo "current_dir: $working_path"
export PYTHONPATH="$working_path"

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type fl \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/goodtriever-gpt2-l-test-${data_type}-8k.jsonl \
       --batch_size 128

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type sim \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/goodtriever-gpt2-l-test-${data_type}-8k.jsonl \
       --batch_size 128