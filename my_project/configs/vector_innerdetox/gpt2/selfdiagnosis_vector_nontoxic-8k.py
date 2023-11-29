_base_ = [
    '../../_base_/p_common.py',
]
prompts_file = "/media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-test-nontoxic-8k.jsonl"

model_type = 'selfdiagnosis_vector_innerdetox'

model_path = '/media/data/1/yx/data/model_cache/gpt2-large'
batch_size = 24  # negative samples 3 --> 24, negative samples 6 --> 16

neg_prompt_idx = 1  # np
pos_prompt_idx = 1  # pp

innerdetox_hook = dict(
    type='BaseInnerDetoxHook',
    norm_exp=0.4,  # ne
    neg_sim_exp=0.6,  # nse
    renorm=True,
    vector_method='mergingtopk50_normmass_dis-max'  # mergingtopk50_normmass_dis-max; mean ;  20_mass_dis-mean ;
)

prompt_type = "nontoxic"
p_num = 3  # 负 + 正 对比样本的数量
pad_neg_sample = True  # 是否使用 padding 的负样本

# todo: method 1. toxicity选择
select_type = "toxicity_topk"
toxicity_attribute = ['toxicity']
