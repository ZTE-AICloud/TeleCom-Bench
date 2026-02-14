import csv
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from .base import BaseDataset


def strip_values(item):
    # 遍历字典的所有项
    for key, value in item.items():
        # 检查值是否为字符串类型
        if isinstance(value, str):
            # 对字符串执行strip操作并更新字典
            item[key] = value.strip()
    return item


@LOAD_DATASET.register_module()
class KECDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # 读取 CSV 文件的第一行（表头行）
            raw_data = []
            for row in reader:  # 使用 for 循环遍历 reader 对象，从第二行开始逐行读取 CSV 文件中的每一行数据
                item = dict(zip(header, row))  # 将当前行 row 与表头 header 对应的列名配对为字典
                item = strip_values(item)
                raw_data.append(item)  # 将处理后的有效数据条目 item 添加到 raw_data 列表中
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class JiebaRouge1Evaluator(JiebaRougeEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        scores = super().score(predictions, references)
        return {
            'rouge1': scores['rouge1']
        }


def KEC_qa_postprocess(text):
    text = text.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    return text
