prompts_file = "/media/data/2/yx/model_toxic/data/ContextLevelToxicity/Toxic_questions.json"

generation_config = {'max_new_tokens': 30,
                     'min_new_tokens': 5,
                     'top_p': 0.9,
                     'do_sample': True}

model_path = "/media/data/1/yx/data/model_cache/llama2-7b"
ppl_model_path = '/media/data/1/yx/data/model_cache/llama2-13b'


DEBIASING_PREFIXES = [
    "The following text contains rude, disrespectful, or unreasonable language:\n",
    "The following text contains very hateful, aggressive, disrespectful language:\n",
    "The following text contains sexually explicit language:\n",
    "The following text contains a threat:\n",
    "The following text contains negative or hateful language targeting someone because of their identity:\n",
    "The following text contains swear words, curse words, or other obscene or profane language:\n"
]

model_type = 'llama2-sd'
seed = 2023
num_k_samples = 25

batch_size = 12
ppl_batch_size = 16
toxicity_batch_size = 64



decay_constant = 50
epsilon = 0.01