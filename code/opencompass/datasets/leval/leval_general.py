import os

from datasets import Dataset, DatasetDict
from jsonlines import jsonlines
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LEvalGeneralDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        split = 'test'
        raw_data = []
        filename = os.path.join(path, split, f'{name}.jsonl')
        with open(filename, 'r', encoding='utf-8') as file:
            for row in jsonlines.Reader(file):
                instructions = row.get('instructions', [])
                outputs = row.get('outputs', [])
                context = row.get('input', [])
                for question, answer in zip(instructions, outputs):
                    raw_data.append({
                        'question': question,
                        'context': context,
                        'length': len(answer.split()),
                        'answer': answer
                    })
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
