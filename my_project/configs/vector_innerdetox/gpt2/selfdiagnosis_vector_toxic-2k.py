_base_ = [
    '../../_base_/p_common.py',
]
prompts_file = "/media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-test-toxic-2k.jsonl"

model_type = 'selfdiagnosis_vector_innerdetox'

model_path = '/media/data/1/yx/data/model_cache/gpt2-large'
ppl_model_path = '/media/data/1/yx/data/model_cache/gpt2-xl'

org_gen_batch_size = 64
scorer_batch_size = 12
batch_size = 24
ppl_batch_size = 16
toxicity_batch_size = 64

neg_prompt_idx = 1  # np
pos_prompt_idx = 1  # pp
pad_neg_sample = True  # 是否使用 padding 的负样本

innerdetox_hook = dict(
    type='BaseInnerDetoxHook',
    norm_exp=0.1,  # ne
    neg_sim_exp=0.1,  # nse
    renorm=True,
    vector_method='mergingtopk50_normmass_dis-max'  # mergingtopk50_normmass_dis-max; mean ;  20_mass_dis-mean ;
)

prompt_type = "toxic"
p_num = 6  # 负 + 正 对比样本的数量

# todo: method 1. toxicity选择
select_type = "toxicity_topk"  # 选择毒性中的随机选择
toxicity_attribute = ['toxicity']
