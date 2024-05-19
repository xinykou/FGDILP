"""
This script is used to transfer the orginal generations to non-toxic generations.
"""

import argparse
import os.path as osp
import sys
import os
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import random
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jsonlines as jsl
import torch
from mmengine import Config
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          GPT2LMHeadModel, GenerationConfig, set_seed, LlamaTokenizer, LlamaForCausalLM)
from Baseline.utils import batchify


def run_appdia_or_paradetox(config, prompts=None, continuations=None):

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = GPT2LMHeadModel.from_pretrained(config.model_path)
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.to('cuda')
    model.eval()

    config.generation_config['eos_token_id'] = model.config.eos_token_id

    generations = []
    pbar = tqdm(batchify(continuations, config.batch_size), total=len(continuations), desc='Generating')
    for contins in pbar:
        contins = [c + ' ' + tokenizer.eos_token for c in contins]
        inputs = tokenizer(contins, return_tensors='pt', padding=True)
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
        pbar.update(len(contins))
    pbar.close()
    return generations
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/media/data/2/yx/model_toxic/Baseline/configs/style_transfer/appdia.py')
    parser.add_argument("--is_toxictity", action="store_true")
    parser.add_argument("--transfer_name", type=str, default='appdia')
    parser.add_argument('--fn', type=str, default='/media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt')
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    set_seed(config.seed)
    if args.is_toxictity:
        fp = args.transfer_name + '-toxic'
        org_gen_fp = config.org_gens_path['toxic']
        print('toxic')
    else:
        fp = args.transfer_name + '-nontoxic'
        org_gen_fp = config.org_gens_path['nontoxic']
        print('nontoxic')

    results_dir = args.fn
    output_fp = osp.join(results_dir, fp + '.jsonl')

    if args.ablation:
        continuations = []
        prompts = []
        data = list(jsl.open(org_gen_fp))
        selected_indices = random.sample(range(len(data)), args.eval_num)  # 选择相同的位置
        for index in selected_indices:
            prompts.append(data[index]['prompt']['text'])
            cont_li = data[index]['continuations']
            continuations.extend([c['text'] for c in cont_li])

    else:
        data = list(jsl.open(org_gen_fp))
        prompts = [d['prompt']['text'] for d in data]
        continuations = []
        for d in data:
            cont_li = d['continuations']
            continuations.extend([c['text'] for c in cont_li])

    if (args.transfer_name == 'appdia' or
            args.transfer_name == 'paradetox' or
            args.transfer_name == 'count'):
        generations = run_appdia_or_paradetox(config, prompts=None, continuations=continuations)
    else:
        raise ValueError('Invalid transfer name')

    result = []
    for i in range(len(prompts)):
        result.append(
            dict(
                prompt=dict(text=prompts[i]),
                continuations=[
                    dict(text=g)
                    for g in generations[
                             i * config.num_k_samples: (i + 1) * config.num_k_samples
                             ]
                ],
            )
        )
    # print("----20个采样样本一行存储完成----")
    jsl.Writer(open(output_fp, 'w', encoding='utf-8')).write_all(result)
