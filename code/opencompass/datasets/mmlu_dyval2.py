import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUDyval2Dataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            print(filename)
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    print(row)
                    print(len(row))
                    assert len(row) == 7
                    raw_data.append({
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'E': row[5],
                        'target': row[6],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset

@LOAD_DATASET.register_module()
class MMLUDyval2LUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            print(filename)
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    print(row)
                    print(len(row))
                    assert len(row) == 7
                    raw_data.append({
                        'input': row[1],
                        'A': row[2],
                        'B': row[3],
                        'C': row[4],
                        'D': row[5],
                        'target': row[6],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset