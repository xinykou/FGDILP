prompts_file = "/media/data/2/yx/model_toxic/data/ContextLevelToxicity/Toxic_questions.json"

generation_config = {'max_new_tokens': 30,
                     'min_new_tokens': 5,
                     'top_p': 0.9,
                     'do_sample': True}

model_path = "/media/data/1/yx/data/model_cache/llama2-7b"
ppl_model_path = '/media/data/1/yx/data/model_cache/llama2-13b'

batch_size = 32
ppl_batch_size = 16
toxicity_batch_size = 64


model_type = 'llama2'
seed = 2023
num_k_samples = 25
