import argparse
import os.path as osp
import os
import pickle as pkl
import sys
from functools import partial
import os
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import jsonlines as jsl
import json
import torch
from mmengine import Config
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, set_seed
from utils import batchify, repeat_interleave

from models.llama.innerdetox_hook import InnerDetoxHook
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from models.llama.modeling_llama_innerdetox import LlamaForCausalLMInnerdetox
import random
from models.sd.modeling import GPT2_or_Llama_Wrapper

def run_innerdetox(config, prompts):
    if 'llama' in config.model_path or \
            'vicuna' in config.model_path:
        tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # model = LlamaForCausalLMInnerdetox.from_pretrained(config.model_path, quantization_config=bnb_config, device_map="auto")
        model = LlamaForCausalLMInnerdetox.from_pretrained(config.model_path, torch_dtype=torch.bfloat16, device_map="auto")

    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.eval()

    neg_prompt = config.neg_prompts
    pos_prompt = config.pos_prompts

    config.generation_config['eos_token_id'] = model.config.eos_token_id

    innerdetox_hook = InnerDetoxHook.build(config.innerdetox_hook)   # todo: 为了记录 最后一个位置
    innerdetox_inputs = dict(
        neg_input_ids=None,
        neg_attention_mask=None,
        innerdetox_hook=innerdetox_hook,
    )

    generations = []
    i = 0
    pbar = tqdm(batchify(prompts, config.batch_size), total=len(prompts), desc=f"Run detoxication with ({config.model_type})")
    for prompt in pbar:
        prompt_w_prefix = [pos_prompt + p for p in prompt] + [
            neg_prompt + p for p in prompt
        ]

        neg_inputs = tokenizer(prompt_w_prefix, padding=True, return_tensors='pt')
        innerdetox_inputs['neg_input_ids'] = neg_inputs['input_ids'].to('cuda')
        innerdetox_inputs['neg_attention_mask'] = neg_inputs['attention_mask'].to(
            'cuda'
        )

        colon_id = tokenizer(neg_prompt.strip())['input_ids'][-1]
        prompt_end_indices = torch.argmax(
            (neg_inputs['input_ids'] == colon_id).long(), dim=1
        )

        old_read_hook = partial(innerdetox_hook.read_hook)
        innerdetox_hook.read_hook = partial(
            innerdetox_hook.read_hook, prompt_end_indices=prompt_end_indices
        )

        inputs = tokenizer(prompt, padding=True, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        generation = model.generate(
            **inputs,
            generation_config=GenerationConfig(**config.generation_config),
            pad_token_id=tokenizer.eos_token_id,
            innerdetox_inputs=innerdetox_inputs,
        )

        innerdetox_hook.read_hook = old_read_hook

        prompt_len = inputs['input_ids'].shape[1]
        generation = tokenizer.batch_decode(
            generation[:, prompt_len:], skip_special_tokens=True
        )
        generations.extend(generation)
        pbar.update(len(prompt))
        i += len(prompt)

    return generations


def run_org_generations(config, prompts):
    if 'llama' in config.model_path or \
            'vicuna' in config.model_path:
        tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # model = LlamaForCausalLM.from_pretrained(config.model_path, quantization_config=bnb_config, device_map="auto")
        model = LlamaForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.bfloat16, device_map="auto")

    else:
        raise ImportError

    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.eval()

    config.generation_config['eos_token_id'] = model.config.eos_token_id
    prompt_prefix = config.get('prompt_prefix', '')

    generations = []
    pbar = tqdm(batchify(prompts, config.batch_size), total=len(prompts), desc=f"Run ({config.model_type}) for org generations")
    for prompt in pbar:
        prompt = [prompt_prefix + p for p in prompt]
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        generation = model.generate(
            **inputs,
            generation_config=GenerationConfig(**config.generation_config),
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs['input_ids'].shape[1]
        generation = tokenizer.batch_decode(
            generation[:, prompt_len:], skip_special_tokens=True
        )

        generations.extend(generation)
        pbar.update(len(prompt))
    return generations

def run_sd(config, prompts):
    wrapper = GPT2_or_Llama_Wrapper(model_name=config.model_path)
    debiasing_prefixes = config.DEBIASING_PREFIXES
    generations = []
    if 'batch_size' not in config:
        batch_size = 1
    else:
        batch_size = config.batch_size
    pbar = tqdm(batchify(prompts, batch_size), total=len(prompts), desc="run generations by sd")
    for text in pbar:
        generations += wrapper.generate_self_debiasing(
            text, debiasing_prefixes=debiasing_prefixes, decay_constant=config.decay_constant,
            epsilon=config.epsilon,
            debug=False, min_length=config.generation_config['min_new_tokens'],
            max_length=config.generation_config['max_new_tokens'],
            do_sample=config.generation_config['do_sample'],
            top_p=config.generation_config['top_p']
        )
        pbar.update(len(text))
    return generations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/sd/sd-llama2-chat-ContextLevel.py",  #
                        type=str)
    parser.add_argument("--fn",
                        type=str,
                        default="/media/data/2/yx/model_toxic/Baseline/results/ContextLevelToxicity/sd-llama2-chat-ContextLevel")
    parser.add_argument("--pre_diagnosis_num",
                        type=int,
                        default=100)
    parser.add_argument("--eval_num",
                        type=int,
                        default=100)

    args = parser.parse_args()

    config = Config.fromfile(args.config)
    config_dict = config.to_dict()
    set_seed(config.seed)

    fn = args.fn
    os.makedirs(fn, exist_ok=True)

    fp = ".".join(osp.basename(args.config).split('.')[:-1])

    in_dir = osp.join(args.fn, fp + '.jsonl')
    if 'innerdetox_hook' in config:
        if 'vector_method' in config.innerdetox_hook:
            vector_method = config.innerdetox_hook['vector_method']
            output_fp = f'{args.fn}/{fp}({vector_method}).jsonl'
        else:
            output_fp = in_dir
    else:
        output_fp = in_dir

    # print(f'Running generation on {args.config} ...')
    print(f'config: --------->')
    for key, value in config_dict.items():
        print(key, value)

    # 1. prompt 获取
    try_prompts = []
    prompts = []
    # ContextLevelToxicity dataset
    if "ContextLevelToxicity" in fn:
        with open(config.prompts_file, 'r') as file:
            data = json.load(file)
            for d in data[args.pre_diagnosis_num:]:  # todo: 开头的部分数据用来评估， 后面部分才是真正测试
                query = d['query']
                question = query.split("Here's the first question: ")[1]
                try_prompts.append(question)

        # random部分生成，放弃全部生成
        selected_indices = random.sample(range(len(try_prompts)), args.eval_num)  # 选择相同的位置
        for index in selected_indices:
            prompts.append(try_prompts[index])
    # RealToxicityPrompt dataset
    else:
        data = jsl.open(config.prompts_file)
        prompts = [d['prompt']['text'] for d in data]

    print("----采样样本存储完成,前5个样例是: ----")
    for item in prompts[:5]:
        print(item)

    if config.model_type == 'llama2-innerdetox' or \
       config.model_type == 'llama2-chat-innerdetox' or \
       config.model_type == 'vicuna-innerdetox':
        generations = run_innerdetox(config, list(repeat_interleave(prompts, config.num_k_samples)))

    elif config.model_type == "llama2-sd" or \
            config.model_type == "llama2-chat-sd" or \
            config.model_type == "vicuna-sd":
        generations = run_sd(config, list(repeat_interleave(prompts, config.num_k_samples)))


    elif config.model_type == 'llama2' or \
        config.model_type == 'llama2-chat' or \
        config.model_type == 'vicuna':
        generations = run_org_generations(config, list(repeat_interleave(prompts, config.num_k_samples)))

    else:
        raise NotImplementedError

    # 2. fueezen model 生成的结果
    result = []

    for i in range(len(prompts)):
        result.append(
            dict(
                prompt=dict(text=prompts[i]),
                continuations=[
                    dict(text=g)
                    for g in generations[
                        i * config.num_k_samples : (i + 1) * config.num_k_samples
                    ]
                ],
            )
        )
    print("----采样样本一行存储完成----")
    jsl.Writer(open(output_fp, 'w', encoding='utf-8')).write_all(result)
