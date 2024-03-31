export CUDA_VISIBLE_DEVICES=0


#python /media/data/2/yx/model_toxic/Baseline/run_generation-gpt2.py \
#    --config configs/atcon/atcon-gpt2-l_rtp-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/atcon/atcon-gpt2-l-rtp-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/atcon/atcon-gpt2-l_rtp-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type ppl
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
#    --config configs/atcon/atcon-gpt2-l_rtp-test-toxic-2k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --ppl_type gpt2-xl \
#    --toxicity_type toxicity



python /media/data/2/yx/model_toxic/Baseline/run_generation-gpt2.py \
    --config configs/atcon/atcon-gpt2-l_rtp-test-nontoxic-8k.py \
    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt


python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
    --config configs/atcon/atcon-gpt2-l_rtp-test-nontoxic-8k.py \
    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
    --eval_type toxicity


python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
    --config configs/atcon/atcon-gpt2-l_rtp-test-nontoxic-8k.py \
    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
    --eval_type ppl


python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
    --config configs/atcon/atcon-gpt2-l_rtp-test-nontoxic-8k.py \
    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
    --ppl_type gpt2-xl \
    --toxicity_type toxicity