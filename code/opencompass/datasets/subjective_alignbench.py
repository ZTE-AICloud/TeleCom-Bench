import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SubjectiveAlignBenchDataset(BaseDataset):

    def load(self, path: str, name: str):
        print("@@@@ SubjectiveAlignBenchDataset begin")
        filename = osp.join(path, f'{name}')
        print("filename is ", filename)
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r') as file:
            for line in file:
                problem = json.loads(line)
                question = problem['question']
                capability = problem['category']
                reference = problem["reference"]
                others = ""
                raw_data.append({
                    'question': question,
                    'others': {
                        'capability': capability
                    },
                    'judge': {
                        'capability': capability,
                        'reference': reference
                    },
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
