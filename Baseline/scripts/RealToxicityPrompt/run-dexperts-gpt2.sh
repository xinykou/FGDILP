export CUDA_VISIBLE_DEVICES=0

# -------------toxic----------------------------------------------------------------------
data_type="toxic"
#python /media/data/2/yx/model_toxic/Baseline/run_generation-gpt2.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt


#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type ppl
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-toxic-2k.py  \
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
#       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/dexperts-gpt2-l_rtp-test-${data_type}-2k.jsonl \
#       --batch_size 128
#
#python ./Baseline/run_evaluation_sim_and_fl.py \
#       --eval_type sim \
#       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
#       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/dexperts-gpt2-l_rtp-test-${data_type}-2k.jsonl \
#       --batch_size 128

# -----------------nontoxic------------------------------------------------
data_type="nontoxic"
#python /media/data/2/yx/model_toxic/Baseline/run_generation-gpt2.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type ppl
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
#    --config configs/dexperts/dexperts-gpt2-l_rtp-test-nontoxic-8k.py \
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
       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/dexperts-gpt2-l_rtp-test-${data_type}-8k.jsonl \
       --batch_size 128

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type sim \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
       --current_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/dexperts-gpt2-l_rtp-test-${data_type}-8k.jsonl \
       --batch_size 128