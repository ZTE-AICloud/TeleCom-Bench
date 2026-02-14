import csv
import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ZteTeleNerDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = {}
        for split in ['dev', 'val', 'test']:
            with open(osp.join(path, split, f'{name}.csv')) as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    item = dict(zip(header, row))
                    dataset.setdefault(split, []).append(item)
        dataset = {i: Dataset.from_list(dataset[i]) for i in dataset}
        return DatasetDict(dataset)

class ZteTeleNerEvaluator(BaseEvaluator):

    def pred_postprocessor(self, text):
        print(text)

        processed_text = [word.strip() for word in text.split(",")]
        return processed_text

    """Evaluator for 5G NR NER."""
    def score(self, predictions, references):
        humaneval_preds = []
        # create json file in human_eval format
        id2score = {}
        idx = 0
        for preds, refer in zip(predictions, references):
            t_score = 0.0
            cnt = 0
            processed_answer = [word.strip() for word in refer.split(",")]
            processed_pred = self.pred_postprocessor(preds)
            for item in processed_pred:
                if item in processed_answer:
                    cnt += 1
            id2score[idx] = float(cnt) * 100 / len(processed_answer)
            idx = idx + 1
        average_score = sum(id2score.values()) / len(predictions)
        return {'score': average_score, 'details': {}}, predictions, id2score