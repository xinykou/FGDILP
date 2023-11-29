export CUDA_VISIBLE_DEVICES=0

#python /media/data/2/yx/model_toxic/Self_Detoxifying/run_generation-gpt2.py \
#    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-toxic-2k.py
#
#
#python /media/data/2/yx/model_toxic/Self_Detoxifying/run_evaluation.py \
#    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-toxic-2k.py \
#    --eval_type toxicity \
#
#
#python /media/data/2/yx/model_toxic/Self_Detoxifying/run_evaluation.py \
#    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-toxic-2k.py \
#    --eval_type ppl
#
#
#python //media/data/2/yx/model_toxic/Self_Detoxifying/merge_evaluations.py \
#    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-toxic-2k.py


#---------------------------------------------------------------------------------------------------------

python /media/data/2/yx/model_toxic/Self_Detoxifying/run_generation-gpt2.py \
    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-nontoxic-8k.py


python /media/data/2/yx/model_toxic/Self_Detoxifying/run_evaluation.py \
    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-nontoxic-8k.py \
    --eval_type toxicity \


python /media/data/2/yx/model_toxic/Self_Detoxifying/run_evaluation.py \
    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-nontoxic-8k.py \
    --eval_type ppl


python //media/data/2/yx/model_toxic/Self_Detoxifying/merge_evaluations.py \
    --config configs/innerdetox/innerdetox-gpt2-l-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-nontoxic-8k.py


