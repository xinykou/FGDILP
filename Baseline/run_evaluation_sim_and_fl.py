import argparse
import os.path as osp
import sys
import os
from API_evaluations.content_similarity import new_wieting_sim
from API_evaluations.fluency import new_do_cola_eval


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jsonlines as jsl
import torch
from mmengine import Config
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          GPT2LMHeadModel, set_seed, LlamaTokenizer, LlamaForCausalLM)
from utils import batchify


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", type=str, default="sim")
    parser.add_argument("--org_path", type=str,
                        default="/media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/gpt2-RealToxicityPrompt-toxic.jsonl")
    parser.add_argument("--current_path", type=str,
                        default="/media/data/2/yx/model_toxic/Baseline/results/RealToxicityPrompt/appdia-toxic.jsonl")

    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    # results_dir = args.fn
    org_results = list(jsl.open(args.org_path))
    if '__' in args.current_path:
        current_path = args.current_path.replace('__', '(', 1)
        current_path = current_path.replace('__', ')', 1)
    else:
        current_path = args.current_path

    current_results = list(jsl.open(current_path))

    org_continuations = []
    current_continuations = []

    if args.eval_type == "fl":
        for org_result, current_result in zip(org_results, current_results):
            current_continuations.extend([p['text'] for p in current_result['continuations']])

        cola_stats, fl_probs_by_sent = new_do_cola_eval(args, preds=current_continuations)
        cola_acc = sum(cola_stats) / len(current_continuations)

    elif args.eval_type == "sim":
        for org_result, current_result in zip(org_results, current_results):
            org_continuations.extend([p['text'] for p in org_result['continuations']])
            current_continuations.extend([p['text'] for p in current_result['continuations']])

        ref_similarity_by_sent = new_wieting_sim(args, refs=org_continuations, preds=current_continuations)
        sim_metric = ref_similarity_by_sent.mean()

    else:
        raise ValueError(f"Unknown evaluation type: {args.eval_type}")


    print(f"Results---{args.eval_type}: {cola_acc if args.eval_type == 'fl' else sim_metric}")

