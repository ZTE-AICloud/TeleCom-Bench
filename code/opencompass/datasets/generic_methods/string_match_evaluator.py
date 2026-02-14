from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import extract_specified_options


# 字符串判断
@ICL_EVALUATORS.register_module()
class StringMatchChoiceEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            correct = False
            pred = extract_specified_options(pred)
            ref = extract_specified_options(ref)
            if pred.upper() == ref.upper():
                correct = True

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if len(is_correct) > 0 else 0.0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct
            }
        }


# 字符串模糊判断
@ICL_EVALUATORS.register_module()
class StringFuzzyMatchEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            correct = False
            pred = pred.upper()
            ref = ref.upper()
            if pred:
                if pred in ref or ref in pred:
                    correct = True

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if len(is_correct) > 0 else 0.0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct
            }
        }


@ICL_EVALUATORS.register_module()
class StringMatchEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str]) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            correct = False
            if pred:
                correct = (pred.upper() == ref.upper())

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if len(is_correct) > 0 else 0.0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct
            }
        }
