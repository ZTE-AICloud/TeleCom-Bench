import csv
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset


@LOAD_DATASET.register_module()
class SpaceDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path) as f:
            reader = csv.reader(f)
            raw_data = []
            for row in reader:
                question = row[0]
                raw_data.append({'question': question, 'answer': ""})
            dataset['test'] = Dataset.from_list(raw_data)
        return dataset


