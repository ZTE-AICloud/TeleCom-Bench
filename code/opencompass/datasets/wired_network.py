import json
from typing import Any, Dict, List

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class WiredNetworkDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        data: List[Dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for item in raw_data["questions"]:
            qtype, content = next(
                (k, v) for k, v in item.items() if k != "id"
            )
            data.append(
                dict(
                    question=content.get("问题", ""),
                    options=content.get("选项", []),
                    type=qtype,
                    answer=content.get("答案", ""),
                )
            )

        return Dataset.from_list(data)
