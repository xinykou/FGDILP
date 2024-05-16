export CUDA_VISIBLE_DEVICES=0,1

name=sd-llama2-chat-ContextLevel
config_path=configs/sd/${name}.py
fn_path=/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/${name}


#python /media/data/2/yx/model_toxic/Baseline/run_generation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --pre_diagnosis_num 100 \
#    --eval_num 100
#
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_num 100 \
#    --eval_type ppl_llama2-13b-chat \


python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --eval_num 100 \
    --eval_type perspective_api_toxicity


python /media/data/2/yx/model_toxic/Baseline/merge_evaluations_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --ppl_type ppl_llama2-13b-chat \
    --toxicity_type perspective_api_toxicity

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