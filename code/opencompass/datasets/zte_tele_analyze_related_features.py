import csv
import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator
import re
from .base import BaseDataset


@LOAD_DATASET.register_module()
class ZteTeleAnalyzeRelatedFeaturesDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        # 加载特性列表
        with open(osp.join(path, 'features.json')) as f:
            feature_list = json.load(f)

        dataset = {}
        for split in ['dev', 'val', 'test']:
            with open(osp.join(path, split, f'{name}.csv')) as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    item = dict(zip(header, row))
                    item['options'] = feature_list
                    dataset.setdefault(split, []).append(item)
        dataset = {i: Dataset.from_list(dataset[i]) for i in dataset}
        return DatasetDict(dataset)

class ZteTeleAnalyzeRelatedFeaturesEvaluator(BaseEvaluator):

    def pred_postprocessor(self, text):
        # 使用正则表达式匹配[]中包含的内容
        matches = re.findall(r'\[([^\]]*)\]', text)
        
        if not matches:
            return []
        
        # 进一步处理匹配到的内容，去除空格并分割为list
        processed_text = [item.strip().strip('"').strip("'") for match in matches for item in match.split(",")]
        processed_text = set(processed_text)
        processed_text = list(processed_text)
        return processed_text

    """Evaluator for 5G NR Feature Match."""
    def score(self, predictions, references):
        id2score = {}
        idx = 0
        for preds, refer in zip(predictions, references):
            t_score = 0.0
            cnt = 0
            # 答案一定是一个list
            processed_answer = eval(refer)
            processed_pred = self.pred_postprocessor(preds)
            print(f"processed_pred:{processed_pred}")
            print(f"processed_answer:{processed_answer}")
            # recall 打分，计算回答，在答案中的占比
            for item in processed_pred:
                if item in processed_answer:
                    cnt += 1
            id2score[idx] = float(cnt) * 100 / len(processed_answer)
            idx = idx + 1
        average_score = sum(id2score.values()) / len(predictions)
        return {'score': average_score, 'details': {}}, predictions, id2score