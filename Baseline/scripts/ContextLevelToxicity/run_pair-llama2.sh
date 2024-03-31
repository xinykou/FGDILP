export CUDA_VISIBLE_DEVICES=0

config_path='configs/others/llama2-ContextLevel.py'
fn_path='/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/llama2-ContextLevel'

#python /media/data/2/yx/model_toxic/Baseline/run_generation_pair-llama2.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --pre_diagnosis_num 100 \
#    --eval_num 100


python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
    --config configs/others/llama2-ContextLevel.py \
    --fn ${fn_path} \
    --eval_type ppl_llama2-13b \


#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_type perspective_api \


python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --ppl_type ppl_llama2-13b \
    --toxicity_type perspective_api

# --------------------------------------------------------------------------------------
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_type llamaguard_toxicity \


#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --ppl_type ppl_llama2-13b \
#    --toxicity_type llamaguard_toxicity





