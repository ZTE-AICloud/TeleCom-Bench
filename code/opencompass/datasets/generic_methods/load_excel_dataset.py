import csv
import os
import json
import os.path as osp

import pandas as pd
from datasets import Dataset, DatasetDict

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class ExcelDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = []
        csv_filename = os.path.join(path, f'{name}.csv')
        xlsx_filename = os.path.join(path, f'{name}.xlsx')
        if os.path.exists(csv_filename):
            print(f"读取数据集：{csv_filename}")
            with open(csv_filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    item = dict(zip(header, row))
                    dataset.append(item)
        elif os.path.exists(xlsx_filename):
            df = pd.read_excel(xlsx_filename)
            # 将所有列转换为字符串类型，并处理NaN值
            df = df.astype(str)  # 先转换为字符串
            df = df.replace('nan', '')  # 将转换后的'nan'字符串替换为空字符串
            df = df.replace('None', '')  # 将'None'字符串也替换为空字符串
            header = list(df.columns)
            for row in df.itertuples(index=False, name=None):
                item = dict(zip(header, row))
                dataset.append(item)
        else:
            raise FileNotFoundError(f"Can't find {csv_filename} or {xlsx_filename}")

        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class ExcelSplitDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = {}
        for split in ['dev', 'val', 'test']:
            csv_filename = os.path.join(path, split, f'{name}_{split}.csv')
            xlsx_filename = os.path.join(path, split, f'{name}_{split}.xlsx')
            if os.path.exists(csv_filename):
                with open(csv_filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:
                        item = dict(zip(header, row))
                        dataset.setdefault(split, []).append(item)
            elif os.path.exists(xlsx_filename):
                df = pd.read_excel(xlsx_filename)
                header = list(df.columns)
                for row in df.itertuples(index=False, name=None):
                    item = dict(zip(header, row))
                    dataset.setdefault(split, []).append(item)
            else:
                raise FileNotFoundError(f"Can't find {csv_filename} or {xlsx_filename}")

        dataset = {i: Dataset.from_list(dataset[i]) for i in dataset}
        return DatasetDict(dataset)

