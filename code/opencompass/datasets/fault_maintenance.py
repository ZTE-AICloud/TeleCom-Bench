import json
import re

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


def _strip_option_prefix(text: str) -> str:
    """去除选项文本中的 'A. '、'B. ' 等前缀"""
    return re.sub(r'^[A-E]\.\s*', '', text)


@LOAD_DATASET.register_module()
class FaultMaintenanceDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        data = []

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            question = item.get("question", "")
            options = item.get("options", [])

            option_a = _strip_option_prefix(options[0]) if len(options) > 0 else ""
            option_b = _strip_option_prefix(options[1]) if len(options) > 1 else ""
            option_c = _strip_option_prefix(options[2]) if len(options) > 2 else ""
            option_d = _strip_option_prefix(options[3]) if len(options) > 3 else ""
            option_e = _strip_option_prefix(options[4]) if len(options) > 4 else ""

            answer = item.get("answer", "")

            data.append(
                {
                    "question": question,
                    "A": option_a,
                    "B": option_b,
                    "C": option_c,
                    "D": option_d,
                    "E": option_e,
                    "answer": answer,
                }
            )

        return Dataset.from_list(data)
