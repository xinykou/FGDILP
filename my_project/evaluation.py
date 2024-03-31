import argparse
import os.path as osp
import sys
import os
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jsonlines as jsl
import torch
from mmengine import Config
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          GPT2LMHeadModel, set_seed, LlamaTokenizer, LlamaForCausalLM)
from utils import batchify
import random
from API_evaluations.perspective_score import perspective_evaluation
from API_evaluations.llamaGuard_score import LlamaGuard

def token_loss(lm_logits, labels):
    bsz = lm_logits.size(0)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(bsz, -1)
    return loss


def run_ppl_evaluation(prompts, continuations, config=None, is_llama=False):
    model_path = config.ppl_model_path
    if 'llama' in model_path or 'vicuna' in model_path:
        is_llama = True

    if is_llama:
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
    else:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to('cuda')

    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    scores = []
    batch_size = config.ppl_batch_size
    model_name = model_path.split('/')[-1]
    pbar = tqdm(
        batchify(zip(prompts, continuations), batch_size), total=len(continuations), desc=f"ppl evaluation by ({model_name})"
    )
    for sample in pbar:
        p_texts, c_texts = zip(*sample)
        texts = []
        for p, c in sample:
            if p[-1] != ' ':
                c = ' ' + c.lstrip()
            texts.append(p + c)

        p_inputs = tokenizer(p_texts)
        inputs = tokenizer(texts, padding=True, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        labels = inputs['input_ids'].clone()
        # ignore pad token
        labels[labels == tokenizer.pad_token_id] = -100
        # conditional ppl, ignore the prompt
        for i in range(len(p_texts)):
            labels[i, : len(p_inputs['input_ids'][i])] = -100

        with torch.no_grad():
            logits = model(**inputs).logits

        loss = token_loss(logits, labels)
        ppls = (loss.sum(dim=-1) / (loss != 0).sum(dim=-1)).exp().cpu().tolist()

        scores.extend(ppls)
        pbar.update(batch_size)
    pbar.close()
    return scores


def run_toxicity_evaluation(continuations, config=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        '/media/data/1/yx/data/model_cache/deberta-v3-large_toxicity-scorer'
    )
    model.to('cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        '/media/data/1/yx/data/model_cache/deberta-v3-large_toxicity-scorer'
    )

    scores = []
    batch_size = config.toxicity_batch_size
    pbar = tqdm(batchify(continuations, batch_size), total=len(continuations))
    for c_texts in pbar:
        inputs = tokenizer(c_texts, padding=True, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().tolist()

        scores.extend(probs)
        pbar.update(len(c_texts))
    pbar.close()
    return scores


def run_toxicity_llamaguard_evaluation(prompts, continuations, config=None):
    assert len(prompts) == len(continuations)
    scores = []
    batch_size = 1
    scorer = LlamaGuard()
    pbar = tqdm(zip(batchify(prompts, batch_size), batchify(continuations, batch_size)), total=len(continuations),
                desc="toxicity evaluation by llamaguard")
    for prompts_batch, continuations_batch in pbar:
        res = scorer.llamaguard_evaluation(prompts_batch, continuations_batch)
        if 'unsafe' in res:
            scores.append(1)
        elif 'safe' in res:
            scores.append(0)
        else:
            raise ValueError
        pbar.update(1)
    pbar.close()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vector_innerdetox/llama2/selfdiagnosis-subtoxicity_vector-llama2-ContextLevel.py")
    parser.add_argument("--fn", type=str, default="/media/data/2/yx/model_toxic/my_project/results/ContextLevelToxicity/selfdiagnosis-subtoxicity_vector-llama2-ContextLevel")
    parser.add_argument("--eval_type", type=str, default="llamaguard_toxicity")
    parser.add_argument("--eval_num", type=int, default=None)
    args = parser.parse_args()
    print(args)
    config = Config.fromfile(args.config)
    set_seed(config.seed)
    fn = 'results/vector_p_muti_gen' if args.fn is None else args.fn

    fp = ".".join(osp.basename(args.config).split('.')[:-1])
    vector_method = config.innerdetox_hook['vector_method']
    in_dir = f'{fn}/{fp}({vector_method}).jsonl'
    output_fp = in_dir

    try_result = list(jsl.open(in_dir))

    if args.eval_num is not None:
        result = []
        selected_indices = random.sample(range(len(try_result)), args.eval_num)  # 选择相同的位置
        for index in selected_indices:
            result.append(try_result[index])
    else:
        result = try_result

    prompts = []
    continuations = []
    for p in result:
        prompts.extend([p['prompt']['text']] * len(p['continuations']))
        continuations.extend([c['text'] for c in p['continuations']])

    if args.eval_type == "ppl":
        score_key = 'ppl'
        scores = run_ppl_evaluation(prompts, continuations, config=config)
    elif args.eval_type == "toxicity":
        score_key = 'toxicity'
        scores = run_toxicity_evaluation(continuations, config=config)
    elif args.eval_type == "ppl_llama2-13b" or \
            args.eval_type == "ppl_llama2-13b-chat" or \
            args.eval_type == "ppl_vicuna-13b-v1.5":
        score_key = 'ppl'
        scores = run_ppl_evaluation(prompts, continuations, config=config)
    elif args.eval_type == "llamaguard_toxicity":
        score_key = 'toxicity'
        scores = run_toxicity_llamaguard_evaluation(prompts, continuations, config=config)
        output_fp = output_fp.replace('.jsonl', '(llamaguard_toxicity).jsonl')
    elif args.eval_type == "perspective_api_toxicity":
        score_key = 'toxicity'
        tags_file = output_fp.replace('.jsonl', '(perspective_api_toxicity).jsonl')
        print("run_perspective_api_toxicity")
        perspective_evaluation(prompts_file=output_fp, tags_file=tags_file)
        exit()
    else:
        raise NotImplementedError


    for i, s in enumerate(scores):
        result[i // config.num_k_samples]['continuations'][i % config.num_k_samples][
            score_key
        ] = s

    jsl.Writer(open(output_fp, 'w', encoding='utf-8')).write_all(result)
