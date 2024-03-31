_base_ = [
    '../_base_/datasets/rtp-test-nontoxic-8k.py',
    '../_base_/common.py',
]

model_type = 'dexperts'

model_path = '/media/data/1/yx/data/model_cache/gpt2-large'
antiexpert_model = '/media/data/2/yx/model_toxic/Baseline/model_cache/dexperts/experts/toxicity/large/finetuned_gpt2_toxic'


dexperts_alpha = 2.0

batch_size = 84