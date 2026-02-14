import csv

from datasets import DatasetDict, Dataset

from opencompass.datasets import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class NebulabizAbstractDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            raw_data = [{k: row[k] for k in ('question', 'length_num')} for row in reader]
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
            return dataset


@ICL_EVALUATORS.register_module()
class NebulabizAbstractEvaluator(BaseEvaluator):

    def is_pred_intact(self, text: str) -> bool:
        """
            False -- 文本异常，中止、不完整
            True  -- 文本正常，完整
        """
        if not isinstance(text, str):
            return False

        if text.startswith('【') and text.endswith('】'):
            text = text.split('【')[1].strip().split('】')[0].strip()

        if '。' not in text:
            return False

        answer_list = text.split('。')
        last_part = answer_list[-1].strip()

        if last_part == "":
            return True
        elif '！' in last_part:
            return not bool(last_part.split('！')[-1].strip())
        elif '？' in last_part:
            return not bool(last_part.split('？')[-1].strip())
        else:
            return True

    def is_length_valid(self, text: str, length_limit: int) -> bool:
        """
            False -- 文本异常，超过长度限制
            True  -- 文本正常，在长度限制内
        """
        if not isinstance(text, str):
            return False
        return len(text) <= length_limit

    def score(self, predictions, length_num):
        cut_results = []
        length_results = []

        for p, l in zip(predictions, length_num):
            pred = p.strip() if isinstance(p, str) else ""
            length_limit = int(l) if str(l).isdigit() else 250

            is_pred_intact = self.is_pred_intact(pred)
            is_valid_length = self.is_length_valid(pred, length_limit)

            cut_results.append(is_pred_intact)
            length_results.append(is_valid_length)

        # pred_intact_rate = sum(cut_results) / len(cut_results)
        # length_valid_rate = sum(length_results) / len(length_results)
        cut_rate = cut_results.count(False) / len(cut_results) if cut_results else 0
        over_length_rate = length_results.count(False) / len(length_results) if length_results else 0

        return {
            "cut_rate": cut_rate * 100,
            "over_length_rate": over_length_rate * 100
        }, cut_results, length_results
