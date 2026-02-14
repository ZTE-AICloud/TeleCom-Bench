import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset, DatasetDict

from opencompass.datasets import BaseDataset, BaseEvaluator
from opencompass.judge_models import JudgeLlama
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS


@LOAD_DATASET.register_module()
class NebulabizFieldsQADataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        filename = os.path.join(path, f'{name}.csv')
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            raw_data = [{k: row[k] for k in ('question', 'reference')} for row in reader]
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
            return dataset


@ICL_EVALUATORS.register_module()
class NebulabizFieldsQA10Evaluator(BaseEvaluator):
    def _evaluate(self, questions, predictions, references, prompt, judge_model):
        if len(predictions) != len(references):
            return {
                'error': 'prediction and reference have different length'
            }

        # score_count = [0, 0, 0, 0]  # 0-3 分
        valid_count = 0
        result_dict = {}

        def extract_score(judge_message):
            matches = re.findall(r'\\?boxed\{(.*?)\}', judge_message)
            if not matches:
                return None
            last_content = matches[-1]
            # number_match = re.search(r'\b[0-9]\b', last_content)
            number_match = re.search(r'\b(10|[0-9])\b', last_content)
            return number_match.group() if number_match else False

        def process_prompt(question, prediction, reference):
            judge_prompt = prompt.format(question=question, prediction=prediction, reference=reference)
            judge_message = judge_model.chat(judge_prompt)
            extract_point = extract_score(judge_message)
            key = hash(question)
            result_dict[key] = {
                'question': question,
                'prediction': prediction,
                'reference': reference,
                'point': extract_point,
                'judge_message': judge_message
            }
            return extract_point

        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = [
                executor.submit(process_prompt, q, p, r)
                for q, p, r in zip(questions, predictions, references)
            ]

            for future in as_completed(futures):
                point = future.result()
                if point is not None:
                    try:
                        point_int = int(point)
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid score: {point}.")
                        # raise ValueError(f"Invalid score: {point}. Score must be an integer between 0 and 3.")

                    if not (0 <= point_int <= 10):
                        raise ValueError(f"Invalid score: {point}.")
                        # raise ValueError(f"Invalid score: {point}. Score must be 0, 1, 2, or 3.")

                    # score_count[point_int] += 1
                    valid_count += 1

        # details = [result_dict[hash(q)] for q in questions]
        # point_list = [result_dict[hash(q)]['point'] for q in questions]
        point_list = [int(result_dict[hash(q)]['point']) for q in questions if 'point' in result_dict[hash(q)]]
        judge_message_list = [result_dict[hash(q)]['judge_message'] for q in questions]
        # score_rate = (score_count[1] + score_count[2] * 2 + score_count[
        #     3] * 3) / (valid_count * 3) if valid_count > 0 else 0
        total_score = sum(point_list)
        score_rate = (total_score / (valid_count * 10)) if valid_count > 0 else 0


        return {
            'score': score_rate * 100,
            'judge_message_list': judge_message_list,
            'point_list': point_list
        }

    def score(self, questions, predictions, references):
        prompt_path = './data/nebulabiz/fields_qa/fields_qa_prompt_10.txt'
        with open(prompt_path, "r", encoding="utf-8") as f:
            simple_qna_prompt = f.read()
        return self._evaluate(questions, predictions, references, simple_qna_prompt, JudgeLlama())


@ICL_EVALUATORS.register_module()
class NebulabizFieldsQA3Evaluator(BaseEvaluator):
    def _evaluate(self, questions, predictions, references, prompt, judge_model):
        if len(predictions) != len(references):
            return {
                'error': 'prediction and reference have different length'
            }

        score_count = [0, 0, 0, 0]  # 0-3 分
        valid_count = 0
        result_dict = {}

        def extract_score(judge_message):
            matches = re.findall(r'\\?boxed\{(.*?)\}', judge_message)
            if not matches:
                return None
            last_content = matches[-1]
            number_match = re.search(r'\b[0-9]\b', last_content)
            return number_match.group() if number_match else False

        def process_prompt(question, prediction, reference):
            judge_prompt = prompt.format(question=question, prediction=prediction, reference=reference)
            judge_message = judge_model.chat(judge_prompt)
            extract_point = extract_score(judge_message)
            key = hash(question)
            result_dict[key] = {
                'question': question,
                'prediction': prediction,
                'reference': reference,
                'point': extract_point,
                'judge_message': judge_message
            }
            return extract_point

        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = [
                executor.submit(process_prompt, q, p, r)
                for q, p, r in zip(questions, predictions, references)
            ]

            for future in as_completed(futures):
                point = future.result()
                if point is not None:
                    try:
                        point_int = int(point)
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid score: {point}. Score must be an integer between 0 and 3.")

                    if not (0 <= point_int <= 3):
                        raise ValueError(f"Invalid score: {point}. Score must be 0, 1, 2, or 3.")

                    score_count[point_int] += 1
                    valid_count += 1

        # details = [result_dict[hash(q)] for q in questions]
        point_list = [result_dict[hash(q)]['point'] for q in questions]
        judge_message_list = [result_dict[hash(q)]['judge_message'] for q in questions]
        score_rate = (score_count[1] + score_count[2] * 2 + score_count[
            3] * 3) / (valid_count * 3) if valid_count > 0 else 0

        return {
            'score': score_rate * 100,
            'judge_message_list': judge_message_list,
            'point_list': point_list
        }

    def score(self, questions, predictions, references):
        prompt_path = './data/nebulabiz/fields_qa/fields_qa_prompt_3.txt'
        with open(prompt_path, "r", encoding="utf-8") as f:
            simple_qna_prompt = f.read()
        return self._evaluate(questions, predictions, references, simple_qna_prompt, JudgeLlama())
