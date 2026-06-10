import json

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class BasicKnowledgeDataset(BaseDataset):
    """读取 basic_knowledge.json，按 tag1 字段筛选不同类别的评测集。"""

    @staticmethod
    def load(path: str, name: str):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        for item in raw.get('questions', []):
            if item.get('tag1') == name:
                data.append({
                    'question': item.get('question', ''),
                    'A': item.get('A', ''),
                    'B': item.get('B', ''),
                    'C': item.get('C', ''),
                    'D': item.get('D', ''),
                    'answer': item.get('answer', ''),
                })

        return Dataset.from_list(data)
