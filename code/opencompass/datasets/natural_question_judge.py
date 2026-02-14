import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.judge_models import VllmOpenAi
from concurrent.futures import ThreadPoolExecutor, as_completed
from opencompass.registry import ICL_EVALUATORS

singel_score_prompt = """
请根据以下提供的回答和标准答案进行评价，并为回答打一个百分制分数：  
[问题]
{question}

[提供的回答]  
{pred}  

[标准答案]  
{answer}  

评价标准：  
1. **准确性**：回答是否与标准答案保持一致，是否完全准确。  
2. **全面性**：回答是否涵盖了标准答案中的关键信息，是否遗漏重要内容。  
3. **清晰性**：回答的表达是否清楚明了，是否易于理解。  
4. **相关性**：回答是否直接与问题相关，是否有无关信息。  

请比较回答和标准答案，并给出一个简短的评价说明，然后为回答打一个百分制分数。  
输出格式如下：  
评价说明：[您的简短评价]  
分数：[[score]]（范围：0-100）
"""

import re

def extract_score(judge_message):
    """
    Extract the score from the model's response using regular expressions.
    Supports both `n分数：[60]` and `n分数：[[60]]` formats.
    If extraction fails, return None.
    """
    try:
        # Match either `[60]` or `[[60]]` and extract the numeric value
        match = re.search(r'\[\[?([0-9]+)\]?\]', judge_message)
        return int(match.group(1)) if match else None
    except (ValueError, AttributeError):
        return None

class NQJudgeEvaluator(BaseEvaluator):
    def score(self, questions, predictions, references):
        judge_model = VllmOpenAi()
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        total_score = 0
        valid_count = 0
        result_dict = {}

        def process_prompt(question, pred, answer):
            judge_prompt = singel_score_prompt.format(question=question, pred=pred, answer=answer)
            judge_message = judge_model.chat(judge_prompt)
            score = extract_score(judge_message)
            # 使用 question 的哈希值作为字典的 key
            key = hash(question)
            result_dict[key] = {
                'prediction': pred,
                'reference': answer,
                'judgement': judge_message,
                'score': score
            }
            return score

        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = [
                executor.submit(process_prompt, question, pred, answer)
                for question, pred, answer in zip(questions, predictions, references)
            ]

            for future in as_completed(futures):
                score = future.result()
                if score is not None:
                    total_score += score
                    valid_count += 1

        # 按 questions 的哈希顺序整理结果
        details = [result_dict[hash(question)] for question in questions]
        explanations = [result['judgement'] for result in details]

        average_score = total_score / valid_count if valid_count > 0 else 0

        return {
            'average_score': average_score,
            'details': details,
            'explanations': explanations,
            'valid_count': valid_count,
            'total_count': len(predictions)
        }


# Prompt for math evaluation   数学题 裁判 模型 打分
math_judge_prompt = """
请根据以下提供的问题、回答和标准答案进行判断，回答是否正确：
[问题]
{question}

[提供的回答]
{pred}

[标准答案]
{answer}

评价标准：
请回答 “对” 或 “错”, 并解释。
"""

def extract_math_judgment(judge_message):
    """
    Extract judgment from the model's response.
    Returns True for "对", False for "错", and None for invalid responses.
    """
    if "对" in judge_message:
        return True
    elif "错" in judge_message:
        return False
    return None

class NQMathJudgeEvaluator(BaseEvaluator):
    def score(self, predictions, references, questions):
        judge_model = VllmOpenAi()
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different lengths'
            }

        correct_count = 0
        total_count = len(predictions)
        result_dict = {}

        def process_prompt(question, pred, answer):
            judge_prompt = math_judge_prompt.format(question=question, pred=pred, answer=answer)
            judge_message = judge_model.chat(judge_prompt)
            judgment = extract_math_judgment(judge_message)
            # 使用 question 的哈希值作为字典的 key
            key = hash(question)
            result_dict[key] = {
                'prediction': pred,
                'reference': answer,
                'judgement': judge_message,
                'is_correct': judgment
            }
            return judgment

        with ThreadPoolExecutor(max_workers=256) as executor:  # 设置线程数
            futures = [
                executor.submit(process_prompt, question, pred, answer)
                for question, pred, answer in zip(questions, predictions, references)
            ]

            for future in as_completed(futures):
                judgment = future.result()
                if judgment is True:
                    correct_count += 1

        # 按 questions 的哈希顺序整理结果
        details = [result_dict[hash(question)] for question in questions]
        explanations = [result['judgement'] for result in details]

        accuracy = correct_count / total_count if total_count > 0 else 0
        score = accuracy * 100

        return {
            'score': score,
            'accuracy': accuracy,
            'details': details,
            'explanations': explanations,
            'correct_count': correct_count,
            'total_count': total_count
        }
