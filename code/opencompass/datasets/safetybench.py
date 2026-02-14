import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

option_lable = ["A", "B", "C", "D", "E", "F"]


def format_data(obj, language: str):
    result = {}
    result["question"] = obj["question"]
    if language == "zh":
        result["question"] = obj["question"].replace("A", "甲").replace("B", "乙")
    result["options"] = ""
    for i in range(len(obj["options"])):
        result["options"] += option_lable[i] + ". " + obj["options"][i] + "\n"
    result["answer"] = option_lable[obj["answer"]]
    return result

def get_dev_datas(path: str, category: str, language: str):
    filename = osp.join(path, f'dev_{language}.json')
    raw_data = []
    with open(filename, 'r') as f:
        datas = json.load(f)
        for key in datas.keys():
            if key.replace(" ", "") == category:
                category = key
                break
        datas_category = datas[category]
        for obj in datas_category:
            result = format_data(obj, language)
            raw_data.append(result)
    return raw_data

def get_val_datas(path: str, category: str, language: str):
    filename = osp.join(path, f'valid_{language}.json')
    raw_data = [] 
    with open(filename, 'r') as f:
        datas = json.load(f)
        for obj in datas:
            if obj["category"].replace(" ", "") == category:
                result = format_data(obj, language)
                raw_data.append(result)
    return raw_data

@LOAD_DATASET.register_module()
class SafetyBenchDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        category = name.split("-")[0]
        language = name.split("-")[1]
        dev_raw_datas = get_dev_datas(path, category, language)
        val_raw_datas = get_val_datas(path, category, language)
        print(f"valid {category} qestion num: ", len(val_raw_datas))
        print(f"dev {category} qestion num: ", len(dev_raw_datas))
        dataset["val"] = Dataset.from_list(val_raw_datas)
        dataset["dev"] = Dataset.from_list(dev_raw_datas)
        return dataset

def safetybench_postprocess(text: str) -> str:
    for t in text:
        if t in ["A", "B", "C", "D"]:
            return t
    if "yes" in text.lower():
        return "A"
    if "no" in text.lower():
        return "B"
    return ''


def num2letter(num: int) -> str:
    if 0 <= num <= 3:
        return option_lable[num]
    return ''