import csv
import json
import re

from datasets import DatasetDict, Dataset

from opencompass.datasets import BaseDataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class NebulabizTranslateDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            raw_data = [row for row in reader]
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
            return dataset


@ICL_EVALUATORS.register_module()
class NebulabizTranslateEvaluator(BaseEvaluator):
    def get_json(self, question: str) -> str:
        start_index = question.find("输入：") + len("输入：")
        end_index = question.find("# 输出", start_index)
        return question[start_index:end_index].strip()

    def deal_json(self, json_str: str, answer_json: dict) -> tuple[int, str]:
        try:
            json_format = json.loads(json_str)
            for key, value in json_format.items():
                if key in answer_json:
                    if value.strip() != "" and str(answer_json.get(key)).strip() == "":
                        return 0, "翻译内容有空"
                    elif "\\t" in value and "\\t" not in answer_json.get(key):
                        return 0, "目录页码有丢失"
                else:
                    return 0, "翻译段落有丢失"
        except Exception:
            return 2, "error"
        return 1, "通过"

    def cut_json(self, s: str) -> bool:
        try:
            json.loads(s)
        except ValueError:
            return True
        return False

    def contains_chinese(self, s: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fa5]', s))

    def get_en_judge(self, question_json: str, answer: str) -> tuple[int, str]:
        if self.cut_json(answer):
            return 0, "json格式不正确"
        elif self.contains_chinese(answer):
            return 0, "输出中包含中文"
        try:
            return self.deal_json(question_json, json.loads(answer))
        except Exception:
            return 2, "error"

    def get_ch_judge(self, question_json: str, answer: str) -> tuple[int, str]:
        if self.cut_json(answer):
            return 0, "json格式不正确"
        elif not self.contains_chinese(answer):
            return 0, "输出中不包含中文"
        try:
            return self.deal_json(question_json, json.loads(answer))
        except Exception:
            return 2, "error"

    def score(self, predictions: list, references: list, questions: list) -> dict:
        results = []
        for pred, question in zip(predictions, questions):
            pred = pred.strip()
            if not pred:
                results.append({"if_pass": 0, "remark": "模型输出为空"})
                continue
            if pred.startswith("```json") and pred.endswith("```"):
                pred = pred.replace("```json", "").replace("```", "").strip()
            question_json = self.get_json(question)
            if "擅长中文表达" in question or "擅长zh-Hans表达" in question:
                if_pass, remark = self.get_ch_judge(question_json, pred)
            else:
                if_pass, remark = self.get_en_judge(question_json, pred)
            results.append({
                "if_pass": if_pass,
                "remark": remark
            })

        score = sum(1 for r in results if r["if_pass"] == 1) / len(results)
        return {
            "score": score,
            "details": results
        }
