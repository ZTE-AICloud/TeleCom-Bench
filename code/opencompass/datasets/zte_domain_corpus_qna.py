import csv
import re

from datasets import Dataset, DatasetDict

from opencompass.judge_models import JudgeLlama
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from .base import BaseDataset
from .subjective.answer_eval import BaseJudgeEvaluator
from concurrent.futures import ThreadPoolExecutor, as_completed


def strip_values(item):
    return {k: v.strip() if isinstance(v, str) else v for k, v in item.items()}


@LOAD_DATASET.register_module()
class ZteDomainCorpusDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as file:
            # DictReader,字典形式返回每一行的内容 {'name': 'Alice', 'age': '30', 'city': 'New York'}
            reader = csv.DictReader(file)
            raw_data = [strip_values(row) for row in reader]
            filtered_data = [{"question": row["question"], "answer": row["answer"]} for row in raw_data]
            dataset["test"] = Dataset.from_list(filtered_data)
            dataset["train"] = Dataset.from_list(filtered_data)
            return dataset


@ICL_EVALUATORS.register_module()
class ZteDomainCorpusQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.result = {}

    def score(self, predictions, references, questions):
        evaluator = ZteDomainCorpusQAJudge()
        self.result = evaluator.score(predictions, references, questions)

        return self.result


judge_prompt = """
# 考题与考生答案
当前有一个考试问题、该问题的标准答案和一位考生的答案：
考试问题：{question}
标准答案：{reference}
考生答案：{prediction}
# 任务
请分析考生答案包含了多少标准答案的内容，给出包含程度评分。
# 判断依据
1、如果考生答案与标准答案中的关键内容完全不相关，存在事实性或逻辑性错误，与标准答案没有任何重合的内容，请给0分。
2、如果考生答案与标准答案相关但不完全一样，与标准答案只有一部分重合的内容，没有涵盖标准答案的所有内容，此时无论考生补充了多少额外内容，都请只给1分。
3、如果考生答案与标准答案完全一致，或考生答案中的内容完全包含了标准答案中的内容，并进行了更多补充，使答案更完整，请给2分。
4、请注意，考生答案包含了多少标准答案的内容是评价的唯一标准，任何考生补充的额外内容都应该被忽略。
# 输出格式要求
请你按照以下句子格式输出，除了该句子之外不得输出任何多余字符！其中分值是一个纯数字，不要输出‘x分’。
分析过程：...；分值：x。
# 开始，输出句子
"""


class ZteDomainCorpusQAJudge(BaseJudgeEvaluator):
    def score(self, predictions, references, questions):
        return self._evaluate_v2(predictions, references, questions, judge_prompt, JudgeLlama())

    def _evaluate_v2(self,  predictions, references,questions, prompt, judge_model):
        if len(predictions) != len(references):
            return {
                'error': 'prediction and reference have different length'
            }

        score_count = [0, 0, 0]
        valid_count = 0
        total_count = len(predictions)
        result_dict = {}

        def extract_score(judge_message):
            process_message = qa_postprocess(judge_message)
            try:
                match = re.search(r'\s*分值：([0-9])', process_message)
                return int(match.group(1)) if match else None
            except(ValueError, AttributeError):
                return None

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
                'evaluation': judge_message
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
                    valid_count += 1
                    try:
                        score_count[point] += 1
                    except(ValueError, AttributeError):
                        return f"Invalid score: {point}. Score must be 0, 1, or 2."

        details = [result_dict[hash(q)] for q in questions]
        score_rate = (score_count[1] * 0.5 + score_count[2]) / valid_count if valid_count > 0 else 0
        return {
            'score': score_rate * 100,
            'point[0,1,2]': score_count,
            'valid_count': valid_count,
            'total_count': total_count,
            'details': details
        }
def qa_postprocess(text):
    text = text.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    return text


def post_process(text):
    pattern = r'Final\s+Answer\s*:\s*(.*?)(?=\n|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        print("No match found")

    return text


def extract_non_reasoning_content(text):
    """
    Remove content within <think>...</think> tags and retain only the content after </think>.
    """
    # Use regular expression to find the closing </think> tag and keep content after it
    result = re.split(r'</think>', text, maxsplit=1)
    if len(result) > 1:
        return result[1].strip()  # Return content after </think>
    return text  # If </think> is not found, return the original text
