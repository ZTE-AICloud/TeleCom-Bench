import concurrent.futures
import csv
import re

import pandas as pd
from datasets import DatasetDict, Dataset
from tqdm import tqdm

from opencompass.datasets import BaseDataset, BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS


@LOAD_DATASET.register_module()
class NebulabizAbstractFormatDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # raw_data = [row['prompt'] for row in reader]
            raw_data = [{k: row[k] for k in ('question', 'format_type')} for row in reader]
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
            return dataset


@ICL_EVALUATORS.register_module()
class NebulabizAbstractFormatEvaluator(BaseEvaluator):
    def score(self, predictions, format_types):
        def check_format(text):
            patterns = [
                r'^#{1,6}\s',  # Markdown 标题
                r'^\d+\.\s',  # 有序列表
                r'^[-*+]\s',  # 无序列表
                r'^> ',  # 引用
                r'```[\s\S]*?```',  # 代码块，多行
                r'!?\[.*?\]\(.*?\)'  # 链接/图片
            ]
            for pattern in patterns:
                if re.search(pattern, text, re.MULTILINE):
                    return True
            return False

        point_list = []
        for prediction, format_type in zip(predictions, format_types):
            has_format = check_format(prediction)

            if int(format_type) in [2, 3]:  # 需要格式的任务
                point = 1 if has_format else 0
            elif int(format_type) in [1, 4]:  # 不应有格式的任务
                point = 0 if has_format else 1
            else:
                raise ValueError(f"Unknown data type: {format_type}")

            point_list.append(point)

        return {
            'score': sum(point_list) / len(point_list) * 100,
            'point_list': point_list
        }


