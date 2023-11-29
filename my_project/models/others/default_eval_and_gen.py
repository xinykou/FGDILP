from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GenerationConfig
from tqdm import tqdm
import torch
import sys
sys.path.append("/media/data/2/yx/model_toxic/my_project/models/others")
from my_project.utils import batchify, repeat_interleave
from base import Base_M

class Default_GPT(Base_M):
    def __init__(self, config):
        super(Default_GPT, self).__init__(config)

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.model = GPT2LMHeadModel.from_pretrained(self.config.model_path)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.model.config.eos_token_id
        self.model.to('cuda')
        self.model.eval()

    def run_gpt2(self, prompts):

        self.config.generation_config['eos_token_id'] = self.model.config.eos_token_id

        prompt_prefix = self.config.get('prompt_prefix', '')
        generations = []
        pbar = tqdm(batchify(prompts, self.config.batch_size), total=len(prompts), desc='run_gpt2 for org generations')
        for prompt in pbar:
            prompt = [prompt_prefix + p for p in prompt]
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

            generation = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(**self.config.generation_config),
                pad_token_id=self.tokenizer.eos_token_id,
            )

            prompt_len = inputs['input_ids'].shape[1]
            generation = self.tokenizer.batch_decode(
                generation[:, prompt_len:], skip_special_tokens=True
            )

            generations.extend(generation)
            pbar.update(len(prompt))
        # 释放模型资源
        return generations

    # 论文中已有的模型作为加载器
    def run_toxicity_evaluation(self, continuations):

        model = AutoModelForSequenceClassification.from_pretrained(
            '/media/data/1/yx/data/model_cache/deberta-v3-large_toxicity-scorer'
        )
        model.to('cuda')
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            '/media/data/1/yx/data/model_cache/deberta-v3-large_toxicity-scorer'
        )

        scores = []
        batch_size = 256
        pbar = tqdm(batchify(continuations, batch_size), total=len(continuations), desc='run_toxicity_evaluation for generations')
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

    def default_filter(self, results=None, scores=None):
        threshold = 0.5
        sample_dict = dict(
            prompt=dict(),
            continuations_toxic=[],
            continuations_nontoxic=[]
        )
        filter_results = []
        for i, s in tqdm(enumerate(scores), desc='default_filter samples'):

            if s >= threshold:
                sample_dict['prompt'] = results[i // self.config.num_k_samples]['prompt']
                text = results[i // self.config.num_k_samples]['continuations'][i % self.config.num_k_samples]['text']
                sample_dict['continuations_toxic'].append(dict(text=text, toxicity=s))
            elif s < threshold:
                sample_dict['prompt'] = results[i // self.config.num_k_samples]['prompt']
                text = results[i // self.config.num_k_samples]['continuations'][i % self.config.num_k_samples]['text']
                sample_dict['continuations_nontoxic'].append(dict(text=text, toxicity=s))

            if (i + 1) % self.config.num_k_samples == 0:
                filter_results.append(sample_dict)
                sample_dict = dict(
                    prompt=dict(),
                    continuations_toxic=[],
                    continuations_nontoxic=[]
                )
        return filter_results

    def rank_and_select(self, filter_results, p_num):
        # ---- 最终使用的那个 毒性输出/ 非毒性输出 位置： 只使用毒性最大的多个样本 和 毒性最小的那个样本的位置----
        num_toxic_total = 0
        num_nontoxic_total = 0
        num_total = len(filter_results)
        max_toxic_index_list = [dict(ids=None) for _ in range(num_total)]
        min_nontoxic_index_list = [dict(ids=None) for _ in range(num_total)]
        for ids, samp in enumerate(filter_results):
            if len(samp['continuations_nontoxic']) == 0:
                num_toxic_total += 1
                toxicity_list = []
                for k, ss in enumerate(samp['continuations_toxic']):
                    toxicity_list.append(ss['toxicity'])
                # 使用列表解析和排序获取最大其索引
                max_indexs = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)[:p_num]
                max_toxic_index_list[ids]['ids'] = max_indexs

            elif len(samp['continuations_toxic']) == 0:
                num_nontoxic_total += 1
                toxicity_list = []
                for j, s in enumerate(samp['continuations_nontoxic']):
                    toxicity_list.append(s['toxicity'])
                min_indexs = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=False)[:p_num]
                min_nontoxic_index_list[ids]['ids'] = min_indexs

            else:
                toxicity_list = []
                nontoxicity_list = []
                for j, s in enumerate(samp['continuations_toxic']):
                    toxicity_list.append(s['toxicity'])
                for j, s in enumerate(samp['continuations_nontoxic']):
                    nontoxicity_list.append(s['toxicity'])

                if len(samp['continuations_nontoxic']) < p_num:
                    min_indexs = sorted(range(len(nontoxicity_list)), key=lambda i: nontoxicity_list[i], reverse=False)
                    min_nontoxic_index_list[ids]['ids'] = min_indexs
                    max_indexs = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)[:p_num]
                    max_toxic_index_list[ids]['ids'] = max_indexs

                elif len(samp['continuations_toxic']) < p_num:
                    max_indexs = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)
                    max_toxic_index_list[ids]['ids'] = max_indexs
                    min_indexs = sorted(range(len(nontoxicity_list)), key=lambda i: nontoxicity_list[i], reverse=False)[
                                 :p_num]
                    min_nontoxic_index_list[ids]['ids'] = min_indexs

                elif len(samp['continuations_nontoxic']) >= p_num and len(samp['continuations_toxic']) >= p_num:
                    min_indexs = sorted(range(len(nontoxicity_list)), key=lambda i: nontoxicity_list[i], reverse=False)[
                                 :p_num]
                    min_nontoxic_index_list[ids]['ids'] = min_indexs
                    max_indexs = sorted(range(len(toxicity_list)), key=lambda i: toxicity_list[i], reverse=True)[:p_num]
                    max_toxic_index_list[ids]['ids'] = max_indexs

        print(f'----prompt type: {self.config.prompt_type}----')
        print(
            f'total num: {num_total} ||  total toxic num: {num_toxic_total}  || total nontoxic num: {num_nontoxic_total}')

        return self.prefix_build(filter_results, max_toxic_index_list, min_nontoxic_index_list, p_num)


    def delete(self):
        if hasattr(self, 'model'):
            self.model.to('cpu')  # 将模型转移到 CPU
            del self.model  # 从命名空间中删除模型变量


