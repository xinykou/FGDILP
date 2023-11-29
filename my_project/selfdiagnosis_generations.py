"""
每个正样例 拼接一个 高毒性的输出， 设置3个毒性负的样例 并行输入，正样例还是毒性最小的那个
"""
import argparse
import os
import os.path as osp
import pickle as pkl
import sys
from functools import partial

sys.path.append("/media/data/2/yx/model_toxic")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jsonlines as jsl
import torch
from mmengine import Config
from tqdm import tqdm
from transformers import set_seed
from utils import batchify, repeat_interleave


from itertools import chain
from models.others.default_eval_and_gen import Default_GPT
from models.others.self_diagnosis import Self_GPT_Score_Rank
from models.others.detox_gen import Detox_GPT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/vector_p_multi_innerdetox/vector_p_multi_selfdiagnosis_nontoxic-8k.py",
                        type=str)
    parser.add_argument("--fn",
                        default='results/vector_p_muti_selfdiagnosis_gen',
                        type=str)

    args = parser.parse_args()
    config = Config.fromfile(args.config)
    config_dict = config.to_dict()
    p_num = config.p_num   # todo: 选择 负的输出的数量， 用于作为融合的来源
    set_seed(config.seed)
    fn = 'results/vector_p_muti_gen' if args.fn is None else args.fn
    os.makedirs(fn, exist_ok=True)
    output_org_fp = f'{fn}/{config.prompt_type}_org_gen.jsonl'
    output_filter_fp = f'{fn}/{config.prompt_type}_filters_gen.jsonl'

    fp = ".".join(osp.basename(args.config).split('.')[:-1])
    # os.makedirs(fp, exist_ok=True)
    vector_method = config.innerdetox_hook['vector_method']
    output_fp = f'{fn}/{fp}({vector_method}).jsonl'
    print(f'Running generation on {args.config} ...')
    print(f'config: --------->')
    for key, value in config_dict.items():
        print(key, value)

    data = jsl.open(config.prompts_file)
    prompts = [d['prompt']['text'] for d in data]

    # 1. 生成原始的输出
    default_gpt = None
    scorer_ranker = None
    if os.path.exists(output_org_fp):
        result = list(jsl.open(output_org_fp))
    else:
        if default_gpt is None:
            default_gpt = Default_GPT(config)
        result = []
        generations_org = default_gpt.run_gpt2(list(repeat_interleave(prompts, config.num_k_samples)))
        torch.cuda.empty_cache()
        for i in range(len(prompts)):
            result.append(
                dict(
                    prompt=dict(text=prompts[i]),
                    continuations=[
                        dict(text=g)
                        for g in generations_org[
                                 i * config.num_k_samples: (i + 1) * config.num_k_samples
                                 ]
                    ],
                )
            )
        jsl.Writer(open(output_org_fp, 'w', encoding='utf-8')).write_all(result)
        default_gpt.delete()  # 释放模型资源
    print("-----org generation finished !-----")

    # 2. 生成评分 + 过滤分为：continuations_toxic, continuations_nontoxic
    if os.path.exists(output_filter_fp):
        filter_results = list(jsl.open(output_filter_fp))
    else:
        continuations = []
        for p in result:
            continuations.extend([c['text'] for c in p['continuations']])
        if 'selfdiagnosis' in config.model_type:
            other_attributes = config.toxicity_attribute[1:]  # 'toxicity' 除了”toxicity“之外的属性
            scorer_ranker = Self_GPT_Score_Rank(config)
            scores = scorer_ranker.run_toxicity_evaluation_by_self(continuations=continuations)
            other_scores = dict()
            for attribute in other_attributes:
                other_score = scorer_ranker.run_toxicity_evaluation_by_self(continuations=continuations,
                                                                             attribute=attribute)
                other_scores[attribute] = other_score
            filter_results = scorer_ranker.self_filter(results=result, scores=scores, other_scores=other_scores, other_attributes=other_attributes)
        else:
            if default_gpt is None:
                default_gpt = Default_GPT(config)
            scores = default_gpt.run_toxicity_evaluation(continuations)
            filter_results = default_gpt.default_filter(result, scores)
            default_gpt.delete()  # 释放模型资源

        jsl.Writer(open(output_filter_fp, 'w', encoding='utf-8')).write_all(filter_results)
    print('-----toxicity classify finished !-----')

    # 3. 按规则选择 负样本
    if 'selfdiagnosis' in config.model_type:
        if scorer_ranker is None:
            scorer_ranker = Self_GPT_Score_Rank(config)
        prepare_res = scorer_ranker.rank_and_select(filter_results, p_num)
        scorer_ranker.delete()
    else:
        if default_gpt is None:
            default_gpt = Default_GPT(config)
        prepare_res = default_gpt.rank_and_select(filter_results, p_num)

    # 4. self detoxify generations
    prompts = [d['prompt']['text'] for d in prepare_res]
    neg_instructs = [d['instructions']['neg_ins'] for d in prepare_res]
    pos_instructs = [d['instructions']['pos_ins'] for d in prepare_res]
    detox_model = Detox_GPT(config)
    generations = detox_model.run_p_innerdetox(
                         list(repeat_interleave(prompts, config.num_k_samples)),
                         list(repeat_interleave(neg_instructs, config.num_k_samples)),
                         list(repeat_interleave(pos_instructs, config.num_k_samples))
                         )
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
    jsl.Writer(open(output_fp, 'w', encoding='utf-8')).write_all(result)



