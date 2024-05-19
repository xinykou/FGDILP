export CUDA_VISIBLE_DEVICES=0


current_dir=$(cd "$(dirname "$0")" && pwd)

first_levels_up_path="$(dirname "$current_dir")" # 回退一级目录: ./Baseline/scripts/
second_levels_up_path="$(dirname "$first_levels_up_path")" # 回退两级目录 ./Baseline
three_levels_up_path="$(dirname "$second_levels_up_path")" # 回退两级目录 ./
# 进入工作目录: ./llama_factory
working_path="${three_levels_up_path}"

cd "$working_path"

echo "current_dir: $working_path"
export PYTHONPATH="$working_path"


output_path=/media/data/2/yx/model_toxic/data/datastore_for_googtriever/outputs
datastore_path=/media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints
neg_name=vicuna_toxic
pos_name=vicuna_nontoxic
name=goodtriever-vicuna-ContextLevel
config_path=configs/goodtriever/${name}.py
fn_path=/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/${name}


#python /media/data/2/yx/model_toxic/Baseline/models/goodtriever/run_all.py \
#  --output_folder ${output_path}/${name} \
#  --prompts_path /media/data/2/yx/model_toxic/data/ContextLevelToxicity/Toxic_questions.json \
#  --model_name /media/data/1/yx/data/model_cache/vicuna-7b-v1.5 \
#  --batch_size 1 \
#  --knn True \
#  --knn_temp 100 \
#  --max_new_tokens 30 \
#  --lmbda 2.0 \
#  --seed 2023 \
#  --eval_num 100 \
#  --dstore_dir ${datastore_path}/${neg_name} \
#  --other_dstore_dir ${datastore_path}/${pos_name}
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_num 100 \
#    --eval_type ppl_vicuna-13b-v1.5 \
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_num 100 \
#    --eval_type perspective_api_toxicity
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --ppl_type ppl_vicuna-13b-v1.5 \
#    --toxicity_type perspective_api_toxicity

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type fl \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/vicuna-ContextLevel/vicuna-ContextLevel.jsonl \
       --current_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/goodtriever-vicuna-ContextLevel/goodtriever-vicuna-ContextLevel.jsonl \
       --batch_size 128

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type sim \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/vicuna-ContextLevel/vicuna-ContextLevel.jsonl \
       --current_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/goodtriever-vicuna-ContextLevel/goodtriever-vicuna-ContextLevel.jsonl \
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
#    --ppl_type ppl_vicuna-13b-v1.5 \
#    --toxicity_type llamaguard_toxicity