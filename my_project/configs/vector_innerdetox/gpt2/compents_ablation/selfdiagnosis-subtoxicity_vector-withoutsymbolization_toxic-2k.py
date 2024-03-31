_base_ = [
    '../../../_base_/p_common.py',
]
prompts_file = "/media/data/2/yx/model_toxic/data/RealToxicityPrompts/rtp-test-toxic-2k.jsonl"

model_type = 'selfdiagnosis-subtoxicity_vector_innerdetox'

model_path = '/media/data/1/yx/data/model_cache/gpt2-large'
ppl_model_path = '/media/data/1/yx/data/model_cache/gpt2-xl'

org_gen_batch_size = 64
scorer_batch_size = 12
batch_size = 12
ppl_batch_size = 16
toxicity_batch_size = 64

neg_prompt_idx = 1  # np
pos_prompt_idx = 1  # pp
# pad_neg_sample = True  # 是否使用 padding 的负样本

innerdetox_hook = dict(
    type='BaseInnerDetoxHook',
    norm_exp=0.4,  # ne
    neg_sim_exp=0.6,  # nse
    renorm=True,
    vector_method='mergingtopk20_None_dis-max',  # mergingtopk50_normmass_dis-max; mean ;  20_mass_dis-mean ;
)

prompt_type = "toxic"
p_num = 6  # 负 + 正 对比样本的数量
# pad_neg_sample = True  # 是否使用 padding 的负样本
# neg_using_max = True  # 是否使用 max 的负样本作为全部负样本

# todo: method 2. toxicity, 其他毒性评分选择
select_type = "toxicity_top1+subtoxicity"
toxicity_attribute = ['toxicity', 'sexually_explicit', 'threat', 'identity_attack', 'profanity', 'insult']  # toxicity, profanity, insult, threat, identity_attack, sexual_explicit