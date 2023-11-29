from transformers import AutoTokenizer, GenerationConfig
from tqdm import tqdm
from my_project.utils import batchify, repeat_interleave
import torch
from functools import partial


class Detox_GPT():
    def __init__(self, config):
        self.config = config
        if config.model_type == "selfdiagnosis_vector_innerdetox" or "selfdiagnosis-subtoxicity_vector_innerdetox":
            from my_project.models.gpt2.modeling_gpt2_innerdetox import GPT2LMHeadModelInnerdetox
            from my_project.models.gpt2.vector_multi_innerdetox_hook import InnerDetoxHook
            self.GPT2LMHeadModelInnerdetox = GPT2LMHeadModelInnerdetox.from_pretrained(config.model_path)
            self.InnerDetoxHook = InnerDetoxHook

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.model = self.GPT2LMHeadModelInnerdetox.from_pretrained(self.config.model_path)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.model.config.eos_token_id
        self.model.to('cuda:0')
        self.model.eval()

    def run_p_innerdetox(self, prompts, neg_ins, pos_ins):
        target_suffix = "before_mergehead"  # 替换为你想要匹配的后缀
        module_names = []
        # 遍历模型的子模块和它们的名称
        for name, child in self.model.named_modules():
            if name.endswith(target_suffix):
                module_names.append(name)

        neg_prompt = self.config.neg_prompts[self.config.neg_prompt_idx]
        pos_prompt_idx = self.config.get('pos_prompt_idx', None)
        pos_prompt = (
            self.config.pos_prompts[pos_prompt_idx] if pos_prompt_idx is not None else ""
        )

        self.config.generation_config['eos_token_id'] = self.model.config.eos_token_id

        innerdetox_hook = self.InnerDetoxHook.build(self.config.innerdetox_hook)
        innerdetox_inputs = dict(
            neg_input_ids=None,
            neg_attention_mask=None,
            innerdetox_hook=innerdetox_hook,
        )

        generations = []

        pbar = tqdm(total=len(prompts), desc='processing on detoxify')
        for idx, (prompt, neg_, pos_) in enumerate(zip(
                batchify(prompts, self.config.batch_size),
                batchify(neg_ins, self.config.batch_size),
                batchify(pos_ins, self.config.batch_size)
        )):
            neg_dict = dict()
            for i in range(self.config.p_num):
                neg_dict[f'neg_{i}'] = []

            prompt_w_prefix = []
            for index, p in enumerate(prompt):
                prompt_w_prefix += [pos_[index][0] + pos_prompt + p]  # 因为 pos 中每个都是重复的

            for index, p in enumerate(prompt):
                for i in range(self.config.p_num):
                    neg_dict[f'neg_{i}'].append(neg_[index][i] + neg_prompt + p)

            for i in range(self.config.p_num):
                prompt_w_prefix += neg_dict[f'neg_{i}']

            neg_inputs = self.tokenizer(prompt_w_prefix, padding=True, return_tensors='pt')
            innerdetox_inputs['neg_input_ids'] = neg_inputs['input_ids'].to('cuda:0')
            innerdetox_inputs['neg_attention_mask'] = neg_inputs['attention_mask'].to(
                'cuda:0'
            )

            colon_id = self.tokenizer(neg_prompt.strip())['input_ids'][-1]
            prompt_end_indices = torch.argmax(
                (neg_inputs['input_ids'] == colon_id).long(), dim=1
            )

            old_read_hook = partial(innerdetox_hook.read_hook)
            innerdetox_hook.read_hook = partial(
                innerdetox_hook.read_hook, prompt_end_indices=prompt_end_indices,
                neg_num_per_sample=self.config.p_num, ids=idx, module_names_specialized=module_names
            )

            inputs = self.tokenizer(prompt, padding=True, return_tensors='pt')
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

            if self.config.model_type == 'selfdiagnosis_vector_innerdetox' or \
                    self.config.model_type == 'selfdiagnosis-subtoxicity_vector_innerdetox':
                generation = self.model.generate(
                    **inputs,
                    generation_config=GenerationConfig(**self.config.generation_config),
                    pad_token_id=self.tokenizer.eos_token_id,
                    innerdetox_inputs=innerdetox_inputs,
                )

            innerdetox_hook.read_hook = old_read_hook

            prompt_len = inputs['input_ids'].shape[1]
            generation = self.tokenizer.batch_decode(
                generation[:, prompt_len:], skip_special_tokens=True
            )
            generations.extend(generation)
            pbar.update(len(prompt))
            i += len(prompt)

        return generations
