import csv
import json
import os.path as osp

from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS

from .base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class NerZteDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            filename = osp.join(path, f'ner_zte_{split}.csv')
            with open(filename) as f:
                reader = csv.reader(f)
                next(reader)  # 跳过第一行
                raw_data = []
                for row in reader:
                    assert len(row) >= 6
                    raw_data.append(
                        {'id': row[0], 'extractElements': row[1], 'elementValueRange': row[2], 'input': row[3],
                         'output': row[4], 'explanation': row[5]})
                dataset[split] = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class NerZteEvaluator(BaseEvaluator):
    def is_str_value_equal(self, str1, str2):
        if isinstance(str1, str) and isinstance(str2, str):
            return str1 in str2 or str2 in str1
        else:
            return False

    def calculate_similarity_score(self, str1: str, str2: str):
        obj1 = {}
        obj2 = {}
        # 将JSON字符串转换为Python对象
        try:
            obj1 = json.loads(str1)
        except ValueError as e:
            print("error: prediction is not valid json:", str1)
            print(e)
            return 0
        try:
            obj2 = json.loads(str2)
        except ValueError as e:
            print("error: reference is not valid json:", str2)
            print(e)
            return 0

        # 计算键值对不匹配的数量
        mismatch_count = 0
        for key in obj1.keys():
            if key not in obj2.keys():
                print(key, "not is obj2, mismatch_count+1")
                mismatch_count += 1
                continue
            if obj1[key] == obj2[key]:
                continue
            if isinstance(obj1[key], list) and isinstance(obj2[key], list):
                if len(obj1[key]) == 1 and len(obj2[key]) == 1:
                    str1 = obj1[key][0]
                    str2 = obj2[key][0]
                    if isinstance(str1, str) and isinstance(str2, str):
                        if str1 in str2 or str2 in str1:
                            mismatch_count += 0.5
                            print(str1, " and ", str2, " half match, mismatch_count+0.5 in str")
                        else:
                            print(str1, " and ", str2, " not match, mismatch_count+1 in str")
                            mismatch_count += 1
                        continue
                if obj1[key] != obj2[key]:
                    try:
                        if (set(obj1[key]).issubset(obj2[key]) or set(obj2[key]).issubset(obj1[key])) \
                                and (len(obj1[key]) != 0 and len(obj2[key]) != 0):
                            mismatch_count += 0.5
                            print(obj1[key], " and ", obj2[key], " half match, mismatch_count+0.5")
                        else:
                            mismatch_count += 1
                            print(obj1[key], " and ", obj2[key], " not match, mismatch_count+1 in list")
                    except:
                        mismatch_count += 1
                        print(obj1[key], " and ", obj2[key], " not valid dict, mismatch_count+1")
            else:
                mismatch_count += 1
                print(obj1[key], " and ", obj2[key], " not match, mismatch_count+1")
        print("mismatch_count: ", mismatch_count)

        # 计算相似度得分
        max_length = max(len(obj1.keys()), len(obj2.keys()))
        similarity_score = (max_length - mismatch_count) / max_length * 100

        return similarity_score

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        num = len(predictions)
        scores = 0
        for i in range(num):
            score1 = self.calculate_similarity_score(predictions[i], references[i])
            score2 = self.calculate_similarity_score(references[i], predictions[i])
            score = score1 if score1 < score2 else score2
            print("prediction: ", predictions[i])
            print("references: ", references[i])
            print("score: ", score)
            scores += score
        scores = scores / num
        return {'score': scores}
