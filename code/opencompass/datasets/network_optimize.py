import json

from datasets import Dataset

from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class NetOptmDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        data = []

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            question_type = item.get("question_type", "")
            stem = item.get("stem", "")

            options = item.get("options", {})
            option_a = options.get("A", "")
            option_b = options.get("B", "")
            option_c = options.get("C", "")
            option_d = options.get("D", "")
            option_e = options.get("E", "")

            correct_answers = item.get("correct_answers", [])
            answer = "".join(map(str, correct_answers))

            data.append(
                {
                    "question_type": question_type,
                    "stem": stem,
                    "A": option_a,
                    "B": option_b,
                    "C": option_c,
                    "D": option_d,
                    "E": option_e,
                    "answer": answer,
                }
            )

        return Dataset.from_list(data)
