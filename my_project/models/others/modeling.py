from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
from typing import List, Optional, Dict, Any, Tuple
import torch

class GPT2Wrapper():
    def __init__(self, model_name: str = "gpt2-xl", cache_path: str = None, use_cuda: bool = True):

        path = os.path.join(cache_path, model_name) if model_name is not None else cache_path
        self._device = 'cuda' if use_cuda else 'cpu'
        self._tokenizer = GPT2Tokenizer.from_pretrained(path)
        self._model = GPT2LMHeadModel.from_pretrained(path).to(self._device)
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
        kwargs = {'add_prefix_space': True} if isinstance(self, GPT2Wrapper) else {}
        for word in output_choices:
            tokens = self._tokenizer.tokenize(word, **kwargs)
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