import importlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Callable

from opencompass.openicl import BaseEvaluator


class BaseJudgeACCEvaluator(BaseEvaluator):
    def __init__(self, prompt: str = None, judge_model=None):
        self._prompt = prompt
        self._judge_model = judge_model
        super().__init__()

    def _get_prompt(self) -> str:
        return (
            "Question: {question}\n"
            "Prediction: {prediction}\n"
            "Reference: {reference}\n"
            "Is the prediction correct? Answer '1' for yes and '0' for no."
        )

    def _get_judge_model(self):
        if self._judge_model is not None:
            if isinstance(self._judge_model, str):
                try:
                    module_path, class_name = self._judge_model.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    self._judge_model = cls()
                except (ImportError, AttributeError, ValueError) as e:
                    raise ImportError(f"Failed to load judge model: {self._judge_model}. Error: {e}")
            return self._judge_model
        from opencompass.judge_models import VllmOpenAi
        return VllmOpenAi()

    def _extract_judge(self, judge_message: str) -> Optional[bool]:
        if "1" in judge_message:
            return True
        elif "0" in judge_message:
            return False
        return None

    def _postprocess(self, scores: dict) -> dict:
        return scores

    def _process_sample(self, question: str, prediction: str, reference: str):
        judge_prompt = self._get_prompt().format(question=question, prediction=prediction, reference=reference)
        judge_message = self._get_judge_model().chat(judge_prompt)
        judge_result = self._extract_judge(judge_message)

        return {
            'judge_prompt': judge_prompt,
            'judge_message': judge_message,
            'judge_result': judge_result
        }

    def score(self, predictions: List, references: List, questions: List) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        valid_count = 0
        correct_count = 0
        results_order = [None] * len(questions)
        detail_dict = {}

        with ThreadPoolExecutor(max_workers=256) as executor:  # 设置线程数
            futures = {
                executor.submit(self._process_sample, q, p, r): idx
                for idx, (q, p, r) in enumerate(zip(questions, predictions, references))
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results_order[idx] = result
                if result['judge_result'] is not None:
                    valid_count += 1
                    if result['judge_result']:
                        correct_count += 1

        seen_keys = set()
        for r in results_order:  # 获取 result 的所有键
            if r is not None:
                seen_keys.update(r.keys())
        for k in seen_keys:  # 初始化 detail_dict 结构
            detail_dict[k] = []
        for r in results_order:  # result 写进 detail_dict 结构，用于打印详情
            for k in seen_keys:
                if r is not None and k in r:
                    detail_dict[k].append(r[k])
                else:
                    detail_dict[k].append(None)

        accuracy = (correct_count / valid_count) if valid_count > 0 else 0
        result = {
            'accuracy': accuracy * 100,
            'detail_dict': detail_dict
        }

        post_result = self._postprocess(result)
        return post_result


class BaseJudgeScoreEvaluator(BaseEvaluator):
    def __init__(self,
                 score_levels: List[int],
                 prompt: str = None,
                 judge_model=None,
                 score_weight_func: Optional[Callable] = None
                 ):
        self._prompt = prompt
        self._judge_model = judge_model
        self._score_levels = sorted(score_levels)
        self._score_weight_func = score_weight_func or (lambda x: x)
        super().__init__()

    def _get_prompt(self) -> str:
        defaul_prompt = """"# 考题与考生答案
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
        return defaul_prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            if isinstance(self._judge_model, str):
                try:
                    module_path, class_name = self._judge_model.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    self._judge_model = cls()
                except (ImportError, AttributeError, ValueError) as e:
                    raise ImportError(f"Failed to load judge model: {self._judge_model}. Error: {e}")
            return self._judge_model
        from opencompass.judge_models import VllmOpenAi
        return VllmOpenAi()

    def _extract_judge(self, judge_message: str):
        try:
            match = re.search(r'(?:分值：|Score|分值:)\s*(\d+)', judge_message.strip())
            if match:
                score = int(match.group(1))
                if score in self._score_levels:
                    return score
            return None
        except Exception:
            return None

    def _postprocess(self, scores: dict) -> dict:
        return scores

    def _process_sample(self, question: str, prediction: str, reference: str):
        judge_prompt = self._get_prompt().format(question=question, prediction=prediction, reference=reference)
        judge_message = self._get_judge_model().chat(judge_prompt)
        judge_result = self._extract_judge(judge_message)

        return {
            'judge_prompt': judge_prompt,
            'judge_message': judge_message,
            'judge_result': judge_result
        }

    def score(self, predictions: List, references: List, questions: List) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        valid_count = 0
        score_count = {level: 0 for level in self._score_levels}
        results_order = [None] * len(questions)
        detail_dict = {}

        with ThreadPoolExecutor(max_workers=64) as executor:  # 设置线程数
            futures = {
                executor.submit(self._process_sample, q, p, r): idx
                for idx, (q, p, r) in enumerate(zip(questions, predictions, references))
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results_order[idx] = result
                score = result['judge_result']
                if score is not None and score in score_count:
                    score_count[score] += 1
                    valid_count += 1

        seen_keys = set()
        for r in results_order:  # 获取 result 的所有键
            if r is not None:
                seen_keys.update(r.keys())
        for k in seen_keys:  # 初始化 detail_dict 结构
            detail_dict[k] = []
        for r in results_order:  # result 写进 detail_dict 结构，用于打印详情
            for k in seen_keys:
                if r is not None and k in r:
                    detail_dict[k].append(r[k])
                else:
                    detail_dict[k].append(None)
        print('score_levels题量列表：', score_count)
        total_weighted_score = sum(
            self._score_weight_func(score) * count
            for score, count in score_count.items()
        )
        score_rate = (total_weighted_score / (valid_count * self._score_levels[-1])) if valid_count > 0 else 0
        result = {
            'score': score_rate * 100,
            'detail_dict': detail_dict
        }

        post_result = self._postprocess(result)
        return post_result
