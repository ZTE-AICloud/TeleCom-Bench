import json
import re
import os
import csv

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

def contains_only_abcd(string):
    # 检查字符串是否只包含字符 'A', 'B', 'C', 'D'
    return set(string) <= set('ABCD')

def contains_only_rightorwrong(string):
    # 检查字符串是否只包含字符 '对', '错'
    return set(string) <= set('对错')

def answer_is_valid(path: str, answer: str):
    # 根据路径类型（选择题或判断题），调用相应的检查函数验证答案的有效性
    if "选择题" in path:
        return contains_only_abcd(answer)
    if "判断题" in path:
        return contains_only_rightorwrong(answer)
    return False

@LOAD_DATASET.register_module() # 将 CompliancezteDataset 类注册到 LOAD_DATASET 注册表中，使其可以被系统识别和加载。
class CompliancezteDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        print("CompliancezteDataset, path is", path)
        validNum = 0
        invalidNum = 0
        dataset = DatasetDict()
        
        # 打开指定路径下的 train.csv 文件
        with open(path + '/train.csv', 'r') as file:
            reader = csv.reader(file)
            header = next(reader) # 读取 CSV 文件的第一行（表头行）
            raw_data = []
            for row in reader: # 使用 for 循环遍历 reader 对象，从第二行开始逐行读取 CSV 文件中的每一行数据
                item = dict(zip(header, row)) # 将当前行 row 与表头 header 对应的列名配对为字典
                if answer_is_valid(path, item["ans"]) and item["question"] != "":
                    # if "." in item["question"]:
                    #     item["question"] = item["question"].split(".")[1] # 去掉句点之前的部分（应该是序号）
                    raw_data.append(item) # 将处理后的有效数据条目 item 添加到 raw_data 列表中
                    validNum += 1
                else:
                    # print(row)
                    invalidNum += 1
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)

        print("validNum is", validNum)
        print("invalidNum is", invalidNum)
        return dataset

