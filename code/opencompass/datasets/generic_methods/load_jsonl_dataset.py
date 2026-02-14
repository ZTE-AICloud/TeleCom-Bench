import json
import os

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class JsonlDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        data = []
        with open(os.path.join(path, f"{name}.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                data.append(line)
        return Dataset.from_list(data)


@LOAD_DATASET.register_module()
class JsonDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        data = []
        with open(os.path.join(path, f"{name}.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)

