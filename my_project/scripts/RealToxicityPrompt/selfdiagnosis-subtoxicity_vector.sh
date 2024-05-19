# 三个 最大毒性， 一个最小毒性， 利用向量融合，
export CUDA_VISIBLE_DEVICES=0

#------------------------------------toxic---------------------------------------------------------------------
data_type='toxic'
#fn_path="/media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector"
#config_path="configs/vector_innerdetox/gpt2/selfdiagnosis-subtoxicity_vector_toxic-2k.py"
#select_ids_path="${fn_path}/toxic_select_ids.jsonl"
#
#
#if [ -e "$select_ids_path" ]; then
#  echo "文件存在"
#else
#  echo "文件不存在"
#  python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#        --config ${config_path} \
#        --fn ${fn_path}
#fi
#
#
#python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#        --config ${config_path} \
#        --fn ${fn_path}
#
## evaluate metrics
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config ${config_path} \
#      --fn ${fn_path} \
#      --eval_type toxicity
#
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config ${config_path} \
#      --fn ${fn_path} \
#      --eval_type ppl
#
## evaluate results
#python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
#      --fn ${fn_path} \
#      --config ${config_path} \
#      --ppl_type ppl \
#      --toxicity_type toxicity



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
#       --current_path /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector/selfdiagnosis-subtoxicity_vector_${data_type}-2k__mergingtopk50_normmass_dis-max__.jsonl \
#       --batch_size 128
#
#python ./Baseline/run_evaluation_sim_and_fl.py \
#       --eval_type sim \
#       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
#       --current_path /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector/selfdiagnosis-subtoxicity_vector_${data_type}-2k__mergingtopk50_normmass_dis-max__.jsonl \
#       --batch_size 128


#--------------------------------nontoxic----------------------------------------------------------------------------------------
data_type='nontoxic'
#fn_path="/media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector"
#config_path="configs/vector_innerdetox/gpt2/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py"
#select_ids_path="${fn_path}/nontoxic_select_ids.jsonl"
#
#
#if [ -e "$select_ids_path" ]; then
#  echo "文件存在"
#else
#  echo "文件不存在"
#  python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#        --config ${config_path} \
#        --fn ${fn_path}
#fi
#
#
#python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#        --config ${config_path} \
#        --fn ${fn_path}
#
## evaluate metrics
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config ${config_path} \
#      --fn ${fn_path} \
#      --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config ${config_path} \
#      --fn ${fn_path} \
#      --eval_type ppl
#
## evaluate results
#python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
#      --config ${config_path} \
#      --fn ${fn_path} \
#      --ppl_type ppl \
#      --toxicity_type toxicity


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
       --current_path /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector/selfdiagnosis-subtoxicity_vector_${data_type}-8k__mergingtopk50_normmass_dis-max__.jsonl \
       --batch_size 128

python ./Baseline/run_evaluation_sim_and_fl.py \
       --eval_type sim \
       --org_path /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-${data_type}.jsonl \
       --current_path /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector/selfdiagnosis-subtoxicity_vector_${data_type}-8k__mergingtopk50_normmass_dis-max__.jsonl \
       --batch_size 128