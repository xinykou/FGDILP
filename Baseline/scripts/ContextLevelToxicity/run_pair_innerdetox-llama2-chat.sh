export CUDA_VISIBLE_DEVICES=0,1


current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./Baseline/scripts/
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./Baseline
three_levels_up_path="$(dirname "$second_levels_up_path")" # 回退两级目录 ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}"

cd "$working_path"

echo "current_dir: $working_path"
export PYTHONPATH="$working_path"



config_path='configs/innerdetox/innerdetox-llama2-chat-ContextLevel.py'
fn_path='/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/innerdetox-llama2-chat-ContextLevel'




#python /media/data/2/yx/model_toxic/Baseline/run_generation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --pre_diagnosis_num 100 \
#    --eval_num 100


#python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_type ppl_llama2-13b-chat \
#    --eval_num 100
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_type perspective_api_toxicity \
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --ppl_type ppl_llama2-13b-chat \
#    --toxicity_type perspective_api_toxicity


python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type fl \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/llama2-chat-ContextLevel/llama2-chat-ContextLevel.jsonl \
       --current_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/innerdetox-llama2-chat-ContextLevel/innerdetox-llama2-chat-ContextLevel.jsonl \
       --batch_size 128

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type sim \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/llama2-chat-ContextLevel/llama2-chat-ContextLevel.jsonl \
       --current_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/innerdetox-llama2-chat-ContextLevel/innerdetox-llama2-chat-ContextLevel.jsonl \
       --batch_size 128

# --------------------------------------------------------------------------------------
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_type llamaguard_toxicity \
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --ppl_type ppl_llama2-13b-chat \
#    --toxicity_type llamaguard_toxicity





