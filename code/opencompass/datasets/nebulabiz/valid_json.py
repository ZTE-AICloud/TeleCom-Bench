import csv
import json
import re

from datasets import DatasetDict, Dataset

from opencompass.datasets import BaseDataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class NebulabizTestDesignDataset(BaseDataset):
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
class NebulabizValidJsonEvaluator(BaseEvaluator):
    def is_valid_json(self, text: str) -> bool:
        """ 判断模型输出是否为有效 JSON
            False -- 无效 JSON
            True  -- 有效 JSON
        """
        if not isinstance(text, str):
            return False

        # 尝试提取 ```json ``` 包裹内容
        matches = re.findall(r'```json(.*?)```', text, re.DOTALL)
        if len(matches) == 1:
            extract_data = matches[0].strip()
        else:
            extract_data = text.strip()

        # 尝试匹配 {} 或 []
        if '[' in extract_data and ']' in extract_data and '{' not in extract_data.split('[')[0]:
            match = re.search(r'\[.*\]', extract_data, re.DOTALL)
        elif '{' in extract_data and '}' in extract_data and '[' not in extract_data.split('{')[0]:
            match = re.search(r'\{.*\}', extract_data, re.DOTALL)
        else:
            return False

        if match:
            try:
                json.loads(match.group(0))
                return True
            except Exception:
                return False
        return False

    def score(self, predictions, references):
        """对 predictions 中的每一项判断其是否为合法 JSON"""
        results = [self.is_valid_json(pred.strip()) for pred in predictions]
        correct = sum(results)
        return {
            'valid_json_rate': correct / len(predictions),
            'is_valid_json_list': results
        }
