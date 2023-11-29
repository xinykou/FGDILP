from typing import Optional
import os
import jsonlines as jsl
from torch.utils.data import Dataset


def load_text_dataset(
        base_path: str,
        filepath: str,
        max_size: Optional[int],

):
    full_filepath = os.path.join(base_path, filepath)
    with jsl.open(full_filepath) as rd:
        source_texts = []
        prompts = []
        for sample in rd:
            source_texts.append(sample['source_text'])
            prompts.append(sample['prompt'])

    if max_size is not None and 'eval' in filepath:
        source_texts = source_texts[:max_size]
        prompts = prompts[:max_size]

    return source_texts, prompts


class Toxic_Nontoxic_Dataset(Dataset):
    def __init__(self, source_texts, prompts):
        assert len(source_texts) == len(prompts)
        self.source_texts = source_texts
        self.prompts = prompts
        super().__init__()

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'prompts': self.prompts[idx]}
        return item


def make_rl_train_datasets(config):
    data_dict = {}
    for split in ['train', 'dev']:
        if split == 'train':
            filepath = config.train_path
        else:
            filepath = config.eval_path

        source_texts, prompts = load_text_dataset(
            config.base_path,
            filepath,
            config.max_eval_size,
        )
        data_dict[split] = Toxic_Nontoxic_Dataset(source_texts, prompts)

    return data_dict['train'], data_dict['dev']
