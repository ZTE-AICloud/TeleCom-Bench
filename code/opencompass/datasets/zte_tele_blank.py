import csv
import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ZteTeleBlankDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = {}
        for split in ['dev', 'val', 'test']:
            filename = osp.join(path, split, f'{name}.csv')
            print(f"读取数据文件:{filename}")
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                print(f"header:{header}")
                for row in reader:
                    item = dict(zip(header, row))
                    #item.setdefault('explanation', '')
                    item.setdefault('answer', '')
                    dataset.setdefault(split, []).append(item)
        dataset = {i: Dataset.from_list(dataset[i]) for i in dataset}
        return DatasetDict(dataset)

def zte_tele_blank_postprocess(text: str) -> str:
    #print(f"后处理:{text}")
    if ":" in text:
        text = text.split(":")[1]
    if "：" in text:
        text = text.split("：")[1]
    return text