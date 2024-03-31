prompts_file = "/media/data/2/yx/model_toxic/data/ContextLevelToxicity/Toxic_questions.json"

generation_config = {'max_new_tokens': 30,
                     'min_new_tokens': 5,
                     'top_p': 0.9,
                     'do_sample': True}

model_path = "/media/data/1/yx/data/model_cache/llama2-7b"
ppl_model_path = '/media/data/1/yx/data/model_cache/llama2-13b'

neg_prompts = "The following text is abusive, harmful, negative, obscene, racist, rude and toxic: "
pos_prompts = "The following text is kind, polite, positive, respectful and supportive: "


model_type = 'llama2-innerdetox'
seed = 2023
num_k_samples = 25

batch_size = 32
ppl_batch_size = 16
toxicity_batch_size = 64

innerdetox_hook = dict(
    type='BaseInnerDetoxHook',
    norm_exp=0.1,  # ne
    neg_sim_exp=0.2,  # nse
    renorm=True,
)