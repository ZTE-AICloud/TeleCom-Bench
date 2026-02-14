import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from ..base import BaseDataset

@LOAD_DATASET.register_module()
class Text2SQLDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['test']:
            filename = osp.join(path, split, name, "question.json")
            raw_data = []
            with open(filename, encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    question = item["prompt"]
                    answer = item["check"]
                    raw_data.append(dict(question=question, answer=answer))
            dataset[split] = Dataset.from_list(raw_data)
        return dataset