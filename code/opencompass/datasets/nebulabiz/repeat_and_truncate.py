import csv
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import DatasetDict, Dataset

from opencompass.datasets import BaseDataset
from opencompass.judge_models import JudgeLlama
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS


@LOAD_DATASET.register_module()
class NebulabizRepeatTruncateDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        filename = os.path.join(path, f'{name}.csv')
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # raw_data = [{k: row[k] for k in ('prompt', 'type')} for row in reader]
            raw_data = [{"question": row["question"]} for row in reader]
            dataset["test"] = Dataset.from_list(raw_data)
            dataset["train"] = Dataset.from_list(raw_data)
            return dataset


def get_last_sentence(text):
    # 获取最后一段文本
    paragraphs = re.split(r'\n{2,}', text)
    paragraphs_clean = [p.strip() for p in paragraphs if p.strip()]
    last_para = paragraphs_clean[-1] if paragraphs_clean else ""

    # 获取最后一句文本
    sentences = re.split('(?<=[。!?！？])', last_para)
    sentences_clean = [s.strip() for s in sentences if s]
    last_sentence = sentences_clean[-1] if sentences_clean else ""

    return last_sentence


def find_repeat_paragraphs(text):
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    def remove_first_digit(paragraph):
        return re.sub(r'\d', '', paragraph, count=1)

    cleaned_paragraphs = [remove_first_digit(p) for p in paragraphs]

    # 合并每两个段落
    merged_paragraphs = []
    for i in range(0, len(cleaned_paragraphs), 2):
        if i + 1 < len(cleaned_paragraphs):
            merged_paragraph = cleaned_paragraphs[i] + ' ' + cleaned_paragraphs[i + 1]
        else:
            merged_paragraph = cleaned_paragraphs[i]
        merged_paragraphs.append(merged_paragraph)

    # 查找是否有重复的段落
    paragraph_counts = Counter(merged_paragraphs)
    repeat_paragraphs = {paragraph: count for paragraph, count in paragraph_counts.items() if count > 1}

    return repeat_paragraphs


nebulabiz_judge_prompt = {
    "truncate_prompt": "请判断以下语句是否一句完整的话，如果是完整的，请回复是，如果不完整，请回复否。请直接输出答案，不要有任何解释性语言。语句为：{prediction}",
    "repeat_prompt": "请分析以下文本，判断其中是否存在连续循环重复输出的语句。如果存在请回复是，如果不存在请回复否，不要有任何解释性语言。文本为：{prediction}"
}


@ICL_EVALUATORS.register_module()
class NebulabizRepeatTruncateEvaluator(BaseEvaluator):
    def score(self, predictions):
        return self._evaluate(predictions, nebulabiz_judge_prompt, JudgeLlama())

    def _evaluate(self, predictions, prompt, judge_model, ):

        repeat_results = []
        truncate_results = []

        def model_judge(prediction, prompt_type):
            judge_prompt = prompt[prompt_type].format(prediction=prediction)
            judge_message = judge_model.chat(judge_prompt)
            return judge_message

        def repeat_eval(prediction, last_sentence, judge_message) -> bool:
            """
                False -- 文本异常，存在重复
                True  -- 文本正常
            """
            if not isinstance(prediction, str) or prediction.strip() == "":
                return False
            if len(prediction) > 20000 or find_repeat_paragraphs(prediction):
                return False
            if len(last_sentence) > 5 and "是" in judge_message:
                return False
            return True

        def truncate_eval(prediction, last_sentence, judge_message) -> bool:
            """
                False -- 文本异常，存在截断
                True  -- 文本正常
            """
            if not isinstance(prediction, str) or prediction.strip() == "":
                return False
            if len(last_sentence) == 1:
                return False
            if len(last_sentence) < 5 and "否" in judge_message:
                return False
            return True

        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = []

            for i, prediction in enumerate(predictions):
                last_sentence = get_last_sentence(prediction)
                futures.append(executor.submit(
                    lambda p, ls: (
                        repeat_eval(p, ls, model_judge(p, "repeat_prompt")),
                        truncate_eval(p, ls, model_judge(p, "truncate_prompt"))
                    ), prediction, last_sentence))

            for future in as_completed(futures):
                is_repeat, is_truncate = future.result()
                repeat_results.append(is_repeat)
                truncate_results.append(is_truncate)

        repeat_rate = repeat_results.count(False) / len(repeat_results) if repeat_results else 0
        truncate_rate = truncate_results.count(False) / len(truncate_results) if truncate_results else 0

        return {
            "repeat_rate": repeat_rate * 100,
            "truncate_rate": truncate_rate * 100
        }, repeat_results, truncate_results
