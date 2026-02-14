import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class Zte5gDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = {}
        for split in ['dev', 'val']:
            with open(osp.join(path, split, f'{name}.csv')) as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    item = dict(zip(header, row))
                    dataset.setdefault(split, []).append(item)
        dataset = {i: Dataset.from_list(dataset[i]) for i in dataset}
        return DatasetDict(dataset)
