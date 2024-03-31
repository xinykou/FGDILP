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
import json
import random
from itertools import chain
from models.others.default_eval_and_gen import Default_GPT_or_Llama
from models.others.self_diagnosis import Self_GPT_or_Llama_Score_Rank
from models.others.detox_gen import Detox_GPT_or_Llama

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/vector_innerdetox/llama2/fusionposition_ablation/selfdiagnosis-subtoxicity_vector_after-linear_toxic-2k.py",
                        type=str)
    parser.add_argument("--fn",
                        default='/media/data/2/yx/model_toxic/my_project/results/RealToxicityPrompts/selfdiagnosis-subtoxicity_vector-fusionposition-llama/after-linear',
                        type=str)
    parser.add_argument("--pre_diagnosis_num",
                        default=100,
                        type=int)
    parser.add_argument("--eval_num",
                        default=100,
                        type=int)
    parser.add_argument("--ablation",
                        action="store_true")

    parser.add_argument("--ablation_layer",
                        default=None,
                        type=str
                        )

    args = parser.parse_args()
    args.ablation = True
    print(args)
    config = Config.fromfile(args.config)

    if args.ablation_layer is not None:
        if 'ablation_layer' in config.innerdetox_hook:
            config.innerdetox_hook.ablation_layer = args.ablation_layer

    config_dict = config.to_dict()
    p_num = config.p_num   # todo: 选择 负的输出的数量， 用于作为融合的来源
    set_seed(config.seed)
    fn = args.fn
    os.makedirs(fn, exist_ok=True)

    output_org_fp = f'{fn}/{config.prompt_type}_org_gen.jsonl' if config.prompt_type is not None else f'{fn}/org_gen.jsonl'
    output_filter_fp = f'{fn}/{config.prompt_type}_filters_gen.jsonl' if config.prompt_type is not None else f'{fn}/filters_gen.jsonl'
    output_select_ids = f'{fn}/{config.prompt_type}_select_ids.jsonl' if config.prompt_type is not None else f'{fn}/select_ids.jsonl'

    fp = ".".join(osp.basename(args.config).split('.')[:-1])
    if 'innerdetox_hook' in config:
        vector_method = config.innerdetox_hook['vector_method']
        output_fp = f'{fn}/{fp}({vector_method}).jsonl'
    else:
        output_fp = f'{fn}/{fp}.jsonl'
    print(f'Running generation on {args.config} ...')
    print(f'config: --------->')
    for key, value in config_dict.items():
        print(key, value)

    # ContextLevelToxicity dataset
    if "ContextLevelToxicity" in fn:
        try_prompts = []
        prompts = []
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
        data = list(jsl.open(config.prompts_file))
        if args.ablation:
            # random部分生成，放弃全部生成
            prompts = []
            selected_indices = random.sample(range(len(data)), args.eval_num)  # 选择相同的位置
            for index in selected_indices:
                prompts.append(data[index]['prompt']['text'])
        else:
            prompts = [d['prompt']['text'] for d in data]

    # 1. 生成原始的输出
    default_gpt = None
    scorer_ranker = None
    if os.path.exists(output_org_fp):
        result = list(jsl.open(output_org_fp))
    else:
        default_gpt = Default_GPT_or_Llama(config)
        result = []
        generations_org = default_gpt.run_gpt2_or_llama(list(repeat_interleave(prompts, config.num_k_samples)))
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
    print("org generations finished!")

    # 2. 生成评分 + 过滤分为：continuations_toxic, continuations_nontoxic
    if os.path.exists(output_filter_fp):
        filter_results = list(jsl.open(output_filter_fp))
    else:
        continuations = []
        for p in result:
            continuations.extend([c['text'] for c in p['continuations']])
        if 'selfdiagnosis' in config.model_type:
            main_attribute = config.toxicity_attribute[0]  # default is 'toxicity'
            other_attributes = config.toxicity_attribute[1:]  # 'toxicity' 除了”toxicity“之外的属性
            scorer_ranker = Self_GPT_or_Llama_Score_Rank(config)
            scores = scorer_ranker.run_toxicity_evaluation_by_self(prompts=list(repeat_interleave(prompts, config.num_k_samples)),
                                                                   continuations=continuations,
                                                                   batch_size=config.scorer_batch_size,
                                                                   attribute=main_attribute,
                                                                   data_is_ContexLevel=True if "ContextLevelToxicity" in fn else False)
            other_scores = dict()
            for attribute in other_attributes:
                other_score = scorer_ranker.run_toxicity_evaluation_by_self(prompts=list(repeat_interleave(prompts, config.num_k_samples)),
                                                                            continuations=continuations,
                                                                            batch_size=config.scorer_batch_size,
                                                                            attribute=attribute,
                                                                            data_is_ContexLevel=True if "ContextLevelToxicity" in fn else False)
                other_scores[attribute] = other_score
            filter_results = scorer_ranker.self_filter(results=result, scores=scores, other_scores=other_scores,
                                                       attribute=main_attribute, other_attributes=other_attributes,
                                                       data_is_ContexLevel=True if "ContextLevelToxicity" in fn else False)

        else:
            if default_gpt is None:
                default_gpt = Default_GPT_or_Llama(config)
            scores = default_gpt.run_toxicity_evaluation(continuations)
            filter_results = default_gpt.default_filter(result, scores)
            default_gpt.delete()  # 释放模型资源

        jsl.Writer(open(output_filter_fp, 'w', encoding='utf-8')).write_all(filter_results)
    print('-----toxicity classify finished !-----')

    # 3. 按规则选择 负样本
    if os.path.exists(output_select_ids):
        prepare_res = list(jsl.open(output_select_ids))
        print("exits select id finished!")
    elif 'selfdiagnosis' in config.model_type:
        if scorer_ranker is None:
            scorer_ranker = Self_GPT_or_Llama_Score_Rank(config)
        prepare_res = scorer_ranker.rank_and_select(filter_results, p_num, main_attribute=config.toxicity_attribute[0])
        jsl.Writer(open(output_select_ids, 'w', encoding='utf-8')).write_all(prepare_res)
        scorer_ranker.delete()
        print("selfdiagnosis select id finished!")
        exit()
    else:
        if default_gpt is None:
            default_gpt = Default_GPT_or_Llama(config)
        prepare_res = default_gpt.random_select(filter_results, p_num, seed=config.seed)
        jsl.Writer(open(output_select_ids, 'w', encoding='utf-8')).write_all(prepare_res)
        default_gpt.delete()
        print("default rank_and_select finished !")
        exit()

    # 4. self detoxify generations
    prompts = [d['prompt']['text'] for d in prepare_res]
    neg_instructs = [d['instructions']['neg_ins'] for d in prepare_res]
    pos_instructs = [d['instructions']['pos_ins'] for d in prepare_res]
    detox_model = Detox_GPT_or_Llama(config)
    generations = detox_model.run_p_innerdetox(
                         list(repeat_interleave(prompts, config.num_k_samples)),
                         list(repeat_interleave(neg_instructs, config.num_k_samples)),
                         list(repeat_interleave(pos_instructs, config.num_k_samples)),
                         ablation_path=args.fn,
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


