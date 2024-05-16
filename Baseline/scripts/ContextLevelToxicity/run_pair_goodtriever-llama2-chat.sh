export CUDA_VISIBLE_DEVICES=0

output_path=/media/data/2/yx/model_toxic/data/datastore_for_googtriever/outputs
datastore_path=/media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints
neg_name=llama2-chat_toxic
pos_name=llama2-chat_nontoxic
name=goodtriever-llama2-chat-ContextLevel
config_path=configs/goodtriever/${name}.py
fn_path=/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/${name}


python /media/data/2/yx/model_toxic/Baseline/models/goodtriever/run_all.py \
  --output_folder ${output_path}/${name} \
  --prompts_path /media/data/2/yx/model_toxic/data/ContextLevelToxicity/Toxic_questions.json \
  --model_name /media/data/1/yx/data/model_cache/llama2-7b-chat \
  --batch_size 1 \
  --knn True \
  --knn_temp 100 \
  --max_new_tokens 30 \
  --lmbda 2.0 \
  --seed 2023 \
  --eval_num 100 \
  --dstore_dir ${datastore_path}/${neg_name} \
  --other_dstore_dir ${datastore_path}/${pos_name}


#python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
#    --config ${config_path} \
#    --fn ${fn_path} \
#    --eval_num 100 \
#    --eval_type ppl_llama2-13b-chat \


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
#    --ppl_type ppl_llama2-13b-chat \
#    --toxicity_type perspective_api_toxicity

# --------------------------------------------------------------------------------------
python /media/data/2/yx/model_toxic/Baseline/run_evaluation_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --eval_type llamaguard_toxicity \


python /media/data/2/yx/model_toxic/Baseline/merge_evaluations_pair.py \
    --config ${config_path} \
    --fn ${fn_path} \
    --ppl_type ppl_llama2-13b-chat \
    --toxicity_type llamaguard_toxicity