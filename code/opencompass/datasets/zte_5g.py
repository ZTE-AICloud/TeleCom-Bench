import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class Zte5gDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for item in raw_data['questions']:
            if item['source_file'] == name:
                data.append({
                    'question': item['question'],
                    'A': item.get('A', ''),
                    'B': item.get('B', ''),
                    'C': item.get('C', ''),
                    'D': item.get('D', ''),
                    'answer': item['answer'],
                })

        return Dataset.from_list(data)
