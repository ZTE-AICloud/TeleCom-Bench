import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

option_lable = ["A", "B", "C", "D", "E", "F"]

@LOAD_DATASET.register_module()
class SafetyBenchTestDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        filename = osp.join(path, f'test_{name}.json')
        with open(filename, 'r') as f:
            datas = json.load(f)
            raw_data = []
            for obj in datas:
                result = {}
                result["question"] = obj["question"]
                if name == "zh":
                    result["question"] = obj["question"].replace("A", "甲").replace("B", "乙")
                result["options"] = ""
                for i in range(len(obj["options"])):
                    result["options"] += option_lable[i] + ". " + obj["options"][i] + "\n"
                result["answer"] = ""
                raw_data.append(result)
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
        return dataset