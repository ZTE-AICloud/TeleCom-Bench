import json
import re
import os
import csv

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

def contains_only_abcd(string):
    return set(string) <= set('ABCD')

def contains_only_rightorwrong(string):
    return set(string) <= set('对错')

def answer_is_valid(path: str, answer: str):
    if "选择题" in path:
        return contains_only_abcd(answer)
    if "判断题" in path:
        return contains_only_rightorwrong(answer)
    return False

@LOAD_DATASET.register_module()
class CommtechzteDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        print("CommtechzteDataset, path is", path)
        validNum = 0
        invalidNum = 0
        dataset = DatasetDict()
        with open(path + '/train.csv', 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            raw_data = []
            for row in reader:
                item = dict(zip(header, row))
                if answer_is_valid(path, item["ans"]) and item["question"] != "":
                    if "." in item["question"]:
                        item["question"] = item["question"].split(".")[1]
                    raw_data.append(item)
                    validNum += 1
                else:
                    # print(row)
                    invalidNum += 1
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
        print("validNum is", validNum)
        print("invalidNum is", invalidNum)
        return dataset

