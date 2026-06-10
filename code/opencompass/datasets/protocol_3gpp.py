import json

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class Protocol3GPPDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        data = raw.get('questions', [])
        return Dataset.from_list(data)
