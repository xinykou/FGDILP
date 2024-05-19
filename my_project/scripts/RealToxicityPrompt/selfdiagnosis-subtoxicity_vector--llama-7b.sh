# 三个 最大毒性， 一个最小毒性， 利用向量融合，
export CUDA_VISIBLE_DEVICES=0

#python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#        --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_toxic-2k.py \
#        --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompt/selfdiagnosis-subtoxicity_vector-llama

# evaluate metrics
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_toxic-2k.py \
#      --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompt/selfdiagnosis-subtoxicity_vector-llama \
#      --eval_type toxicity


python /media/data/2/yx/model_toxic/my_project/evaluation.py \
      --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_toxic-2k.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompt/selfdiagnosis-subtoxicity_vector-llama \
      --eval_type ppl_llama-13b \
      --eval_num 100

# evaluate results
python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompt/selfdiagnosis-subtoxicity_vector-llama \
      --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_toxic-2k.py \
      --ppl_type ppl_llama-13b \
      --toxicity_type chatgpt_api_toxicity


#------------------------------------------------------------------------------------------------------------------------
#python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
#        --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py \
#        --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis-subtoxicity_vector-llama
#
## evaluate metrics
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py \
#      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis-subtoxicity_vector-llama\
#      --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/my_project/evaluation.py \
#      --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py \
#      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis-subtoxicity_vector-llama \
#      --eval_type ppl_llama-13b \
#      --eval_num 100
#
## evaluate results
#python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
#      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis-subtoxicity_vector-llama \
#      --config configs/vector_innerdetox/llama/selfdiagnosis-subtoxicity_vector_nontoxic-8k.py \
#      --ppl_type ppl_llama-13b