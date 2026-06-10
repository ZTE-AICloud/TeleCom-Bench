import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from datasets import Dataset

from opencompass.datasets import BaseDataset, BaseJudgeACCEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from opencompass.datasets import BaseJudgeScoreEvaluator
from opencompass.judge_models.openai_judge import maybe_build_openai_judge


@LOAD_DATASET.register_module()
class CoreNetwork(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        data = []
        for item in raw.get('questions', []):
            data.append(dict(
                question=item.get('题目', ''),
                answer=item.get('答案', ''),
            ))
        return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class CoreNetworkEvaluator(BaseJudgeACCEvaluator):
    def __init__(self, prompt=None, judge_model=None):
        super().__init__(prompt=prompt, judge_model=judge_model)

    def _get_prompt(self) -> str:
        prompt = """# 考题与考生答案
当前有一个考试问题、该问题的标准答案和一位考生的答案：
考试问题：{question}
标准答案：{reference}
考生答案：{prediction}
# 任务
请分析考生答案与标准答案的最终结果是否一致。
# 判断依据
1、如果考生答案与标准答案中的最终结果不一致，请给0分。
2、如果考生答案与标准答案中的最终结果一致，请给1分。
3、请注意，考生答案与标准答案中最终结果的一致性是评价的唯一标准，任何考生补充的额外内容都应该被忽略。
# 输出格式要求
请你按照以下句子格式输出，除了该句子之外不得输出任何多余字符！其中分值是一个纯数字，不要输出‘x分’。
分析过程：...；分值：x。
# 开始，输出句子"""
        return prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            self._judge_model = maybe_build_openai_judge(self._judge_model)
            return self._judge_model
        from opencompass.judge_models.judge_llama import JudgeLlama

        return JudgeLlama()

    def _extract_judge(self, judge_message: str):
        try:
            match = re.search(r"(?:分值：|分值:)\s*(\d+)", judge_message.strip())
            if match:
                score = int(match.group(1))
                if score == 1:
                    return True
                elif score == 0:
                    return False
                return None
            return None
        except Exception:
            return None
