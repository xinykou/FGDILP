# 三个 最大毒性， 一个最小毒性， 利用向量融合，
export CUDA_VISIBLE_DEVICES=0

python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
        --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_toxic-2k.py \
        --fn /media/data/2/yx/model_toxic/my_project/results/vector_p_muti_selfdiagnosis_gen

# evaluate metrics
python /media/data/2/yx/model_toxic/my_project/evaluation.py \
      --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_toxic-2k.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis_gen \
      --eval_type toxicity


python /media/data/2/yx/model_toxic/my_project/evaluation.py \
      --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_toxic-2k.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis_gen \
      --eval_type ppl

# evaluate results
python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis_gen \
      --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_toxic-2k.py

#------------------------------------------------------------------------------------------------------------------------
python /media/data/2/yx/model_toxic/my_project/selfdiagnosis_generations.py \
        --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_nontoxic-8k.py \
        --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis_gen

# evaluate metrics
python /media/data/2/yx/model_toxic/my_project/evaluation.py \
      --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_nontoxic-8k.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis_gen \
      --eval_type toxicity


python /media/data/2/yx/model_toxic/my_project/evaluation.py \
      --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_nontoxic-8k.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis_gen \
      --eval_type ppl

# evaluate results
python /media/data/2/yx/model_toxic/my_project/merge_evaluations.py \
      --fn /media/data/2/yx/model_toxic/my_project/results/selfdiagnosis_gen \
      --config configs/vector_innerdetox/gpt2/selfdiagnosis_vector_nontoxic-8k.py



