export CUDA_VISIBLE_DEVICES=1

# toxic prompt generate
python /media/data/2/yx/model_toxic/Baseline/models/goodtriever/run_all.py \
  --output_folder /media/data/2/yx/model_toxic/data/datastore_for_googtriever/outputs/goodtriever-large \
  --prompts_path /media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-test-toxic-2k.jsonl \
  --model_name /media/data/1/yx/data/model_cache/gpt2-large \
  --batch_size 4 \
  --knn True \
  --knn_temp 100 \
  --lmbda 2.0 \
  --dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_toxic \
  --other_dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_nontoxic

python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
    --config configs/goodtriever/goodtriever-gpt2-l-test-toxic-2k.py \
    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
    --eval_type toxicity


python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
    --config configs/goodtriever/goodtriever-gpt2-l-test-toxic-2k.py \
    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
    --eval_type ppl


python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
    --config configs/goodtriever/goodtriever-gpt2-l-test-toxic-2k.py \
    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
    --ppl_type gpt2-xl \
    --toxicity_type toxicity

#-----------------------------------------------------------------------------------------------------------------------
# nontoxic prompt generate
#python /media/data/2/yx/model_toxic/Baseline/models/goodtriever/run_all.py \
#  --output_folder /media/data/2/yx/model_toxic/data/datastore_for_googtriever/outputs/goodtriever-large \
#  --prompts_path /media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-test-nontoxic-8k.jsonl \
#  --model_name /media/data/1/yx/data/model_cache/gpt2-large \
#  --batch_size 4 \
#  --knn True \
#  --knn_temp 100 \
#  --lmbda 2.0 \
#  --dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_toxic \
#  --other_dstore_dir /media/data/2/yx/model_toxic/data/datastore_for_googtriever/checkpoints/gpt2-large_nontoxic

#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type toxicity
#
#
#python /media/data/2/yx/model_toxic/Baseline/run_evaluation.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --eval_type ppl
#
#
#python /media/data/2/yx/model_toxic/Baseline/merge_evaluations.py \
#    --config configs/goodtriever/goodtriever-gpt2-l-test-nontoxic-8k.py \
#    --fn /media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt \
#    --ppl_type gpt2-xl \
#    --toxicity_type toxicity

