import json

import re

from typing import List

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def extract_option_number(text):
    match = re.search(r"option\s+(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


@LOAD_DATASET.register_module()
class TeleQnADataset(BaseDataset):
    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            question_bank = json.load(f)

        data = []
        for q_name, item in question_bank.items():
            # 提取option 1, option 2, ...
            option_keys = sorted([k for k in item.keys() if k.startswith("option ")])

            # 构建包含问题和所有option的字典
            question_data = {"question": item.get("question")}
            for opt_key in option_keys:
                question_data[opt_key] = item[opt_key]

            # 生成JSON问题和选项回填prompt模板
            question_json = json.dumps(
                {q_name: question_data}, ensure_ascii=False, indent=4
            )

            data.append(
                {
                    "question_json": question_json,
                    "answer": item.get("answer", ""),
                }
            )

        return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class TeleQnAEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references have different lengths")

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            pred_num = extract_option_number(pred)
            ref_num = extract_option_number(ref)
            correct = (pred_num is not None) and (pred_num == ref_num)

            is_correct.append(correct)
            processed_pred.append(pred_num)
            processed_gold.append(ref_num)

        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct,
            },
        }
