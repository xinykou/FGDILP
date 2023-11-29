import argparse
import os.path as osp
import pickle as pkl
import sys
from functools import partial

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import jsonlines as jsl
import torch
from mmengine import Config
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, set_seed
from utils import batchify, repeat_interleave

from models.llama.innerdetox_hook import InnerDetoxHook
from transformers import LlamaForCausalLM, LlamaTokenizer
from models.llama.modeling_llama_innerdetox import LlamaForCausalLMInnerdetox



def run_innerdetox(config, prompts):
    tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = LlamaForCausalLMInnerdetox.from_pretrained(config.model_path, torch_dtype=torch.float16)
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.to('cuda')
    model.eval()

    neg_prompt = config.neg_prompts[config.neg_prompt_idx]
    pos_prompt_idx = config.get('pos_prompt_idx', None)
    pos_prompt = (
        config.pos_prompts[pos_prompt_idx] if pos_prompt_idx is not None else ""
    )

    config.generation_config['eos_token_id'] = model.config.eos_token_id

    innerdetox_hook = InnerDetoxHook.build(config.innerdetox_hook)   # todo: 为了记录 最后一个位置
    innerdetox_inputs = dict(
        neg_input_ids=None,
        neg_attention_mask=None,
        innerdetox_hook=innerdetox_hook,
    )

    generations = []
    i = 0
    pbar = tqdm(batchify(prompts, config.batch_size), total=len(prompts))
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

    print("-----生成完成------")
    return generations


def run_llama(config, prompts):
    tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = LlamaForCausalLM.from_pretrained(config.model_path,torch_dtype=torch.float16)
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.to('cuda')
    model.eval()

    config.generation_config['eos_token_id'] = model.config.eos_token_id

    prompt_prefix = config.get('prompt_prefix', '')

    generations = []
    pbar = tqdm(batchify(prompts, config.batch_size), total=len(prompts))
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/innerdetox/innerdetox-llama-ne0.4-nse0.6-renorm_np0-pp0_rtp-test-toxic-2k.py",
                        type=str)
    parser.add_argument("--new_model_type",
                        default="innerdetox",   # base_model or innerdetox
                        type=str)
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    config.model_type = args.new_model_type
    set_seed(config.seed)

    fn = ".".join(osp.basename(args.config).split('.')[:-1])
    output_fp = f'results/{fn}.jsonl'

    print(f'Running generation on {args.config} ...')
    print(f'model_type: {config.model_type}')

    data = jsl.open(config.prompts_file)
    prompts = [d['prompt']['text'] for d in data]


    if config.model_type == 'innerdetox':
        gen_fn = run_innerdetox
    elif config.model_type == 'base_model':
        gen_fn = run_llama
    else:
        raise NotImplementedError

    generations = gen_fn(config, list(repeat_interleave(prompts, config.num_k_samples)))

    result = []
    for i in range(len(prompts)):
        print()
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
    # print("----20个采样样本一行存储完成----")
    jsl.Writer(open(output_fp, 'w', encoding='utf-8')).write_all(result)
