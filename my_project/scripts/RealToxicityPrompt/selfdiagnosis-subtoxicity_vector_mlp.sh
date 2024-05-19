# 三个 最大毒性， 一个最小毒性， 利用向量融合，
export CUDA_VISIBLE_DEVICES=1

fn_path="/media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector"
config_path="configs/vector_innerdetox/gpt2/selfdiagnosis-subtoxicity_vector_mlp_toxic-2k.py"
select_ids_path="${fn_path}/toxic_select_ids.jsonl"


if [ -e "$select_ids_path" ]; then
  echo "文件存在"
else
  echo "文件不存在"
  python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
        --config ${config_path} \
        --fn ${fn_path}
fi


python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
        --config ${config_path} \
        --fn ${fn_path}

# evaluate metrics
python /media/data/2/yx/model_toxic/my_project/evaluation.py \
      --config ${config_path} \
      --fn ${fn_path} \
      --eval_type toxicity

python /media/data/2/yx/model_toxic/my_project/evaluation.py \
      --config ${config_path} \
      --fn ${fn_path} \
      --eval_type ppl

# evaluate results
python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
      --config ${config_path} \
      --fn ${fn_path} \
      --ppl_type ppl \
      --toxicity_type toxicity

#------------------------------------------------------------------------------------------------------------------------
#python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#        --config configs/vector_innerdetox/gpt2/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py \
#        --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector
#
## evaluate metrics
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config configs/vector_innerdetox/gpt2/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py \
#      --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector \
#      --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config configs/vector_innerdetox/gpt2/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py \
#      --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector \
#      --eval_type ppl
#
## evaluate results
#python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
#      --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector \
#      --config configs/vector_innerdetox/gpt2/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py