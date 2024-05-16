export CUDA_VISIBLE_DEVICES=0,1

config_path='configs/innerdetox/innerdetox-llama2-chat-ContextLevel.py'
fn_path='/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/innerdetox-llama2-chat-ContextLevel'

#python /media/data/2/yx/model_toxic/Baseline/run_generation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --pre_diagnosis_num 100 \
#    --eval_num 100


python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --eval_type ppl_llama2-13b-chat \
    --eval_num 100


python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --eval_type perspective_api_toxicity \


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





