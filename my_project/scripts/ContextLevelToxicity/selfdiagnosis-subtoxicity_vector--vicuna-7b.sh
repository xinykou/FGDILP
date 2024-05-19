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


name=selfdiagnosis-subtoxicity_vector-vicuna-ContextLevel
config_path=./my_project/configs/vector_innerdetox/llama2/${name}.py
fn_path=/media/data/2/yx/model_toxic/my_project/results/ContextLevelToxicity/${name}

first_select=False


#python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --pre_diagnosis_num 100 \
#    --eval_num 100
#
#if [ "$first_select" = "True" ]; then
#    echo "第一次选择pos 和 neg"
#    python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --pre_diagnosis_num 100 \
#    --eval_num 100
#fi
#
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_num 100 \
#    --eval_type ppl_vicuna-13b-v1.5 \


#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_num 100 \
#    --eval_type perspective_api_toxicity
#
#
#python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --ppl_type ppl_vicuna-13b-v1.5 \
#    --toxicity_type perspective_api_toxicity


python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type fl \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/vicuna-ContextLevel/vicuna-ContextLevel.jsonl \
       --current_path /media/data/2/yx/model_toxic/my_project/results/ContextLevelToxicity/selfdiagnosis-subtoxicity_vector-vicuna-ContextLevel/selfdiagnosis-subtoxicity_vector-vicuna-ContextLevel__mergingtopk50_normmass_dis-max__.jsonl \
       --batch_size 128

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type sim \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/vicuna-ContextLevel/vicuna-ContextLevel.jsonl \
       --current_path /media/data/2/yx/model_toxic/my_project/results/ContextLevelToxicity/selfdiagnosis-subtoxicity_vector-vicuna-ContextLevel/selfdiagnosis-subtoxicity_vector-vicuna-ContextLevel__mergingtopk50_normmass_dis-max__.jsonl \
       --batch_size 128

# --------------------------------------------------------------------------------------
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_type llamaguard_toxicity \


#python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --ppl_type ppl_vicuna-13b-v1.5 \
#    --toxicity_type llamaguard_toxicity