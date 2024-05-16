export CUDA_VISIBLE_DEVICES=0

config_path='configs/innerdetox/innerdetox-vicuna-ContextLevel.py'
fn_path='/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/innerdetox-vicuna-ContextLevel'

python /media/data/2/yx/model_toxic/Baseline/run_generation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --pre_diagnosis_num 100 \
    --eval_num 100


python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --eval_type ppl_vicuna-13b \
    --eval_num 100


python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --eval_type perspective_api_toxicity \
    --eval_num 100


python /media/data/2/yx/model_toxic/Baseline/merge_evaluations_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --ppl_type ppl_vicuna-13b-v1.5 \
    --toxicity_type perspective_api_toxicity

# --------------------------------------------------------------------------------------
python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --eval_type llamaguard_toxicity \
    --eval_num 100


python /media/data/2/yx/model_toxic/Baseline/merge_evaluations_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --ppl_type ppl_vicuna-13b-v1.5 \
    --toxicity_type llamaguard_toxicity





