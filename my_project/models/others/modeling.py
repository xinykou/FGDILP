from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
import os
from typing import List, Optional, Dict, Any, Tuple
import torch

class GPT2_or_Llama_Wrapper():
    def __init__(self, model_name: str = None, cache_path: str = None, use_cuda: bool = True):


        self._device = 'cuda' if use_cuda else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained(cache_path)
        if 'gpt2' in cache_path:
            self._model = GPT2LMHeadModel.from_pretrained(cache_path).to(self._device)

        elif 'llama' in cache_path or 'vicuna' in cache_path:
            self._model = LlamaForCausalLM.from_pretrained(cache_path, torch_dtype=torch.bfloat16, device_map='auto')
        else:
            raise ValueError('model_path must be gpt2 or llama')

        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id


    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs['attention_mask'].sum(dim=1) - 1
        output = self._model(**inputs)['logits']
        return torch.stack([output[example_idx, last_word_idx, :] for example_idx, last_word_idx in enumerate(output_indices)])

    def get_token_probability_distribution(self, input_texts: List[str], output_choices: List[str]) -> List[
        List[Tuple[str, float]]]:
        output_choice_ids = []

        for word in output_choices:
            tokens = self._tokenizer.tokenize(word)
            assert len(tokens) == 1, f"Word {word} consists of multiple tokens: {tokens}"
            assert tokens[0] not in self._tokenizer.all_special_tokens, f"Word {word} corresponds to a special token: {tokens[0]}"
            token_id = self._tokenizer.convert_tokens_to_ids(tokens)[0]
            output_choice_ids.append(token_id)

        logits = self.query_model_batch(input_texts)  # 得到 ”yes“ 和 ”no“的概率
        result = []

        for idx, _ in enumerate(input_texts):
            output_probabilities = logits[idx][output_choice_ids].softmax(dim=0)
            choices_with_probabilities = list(zip(output_choices, (prob.item() for prob in output_probabilities)))
            result.append(choices_with_probabilities)

        return result

    def delete(self):
        if hasattr(self, '_model'):
            self._model.to('cpu')  # 将模型转移到 CPU
            del self._model  # 从命名空间中删除模型变量