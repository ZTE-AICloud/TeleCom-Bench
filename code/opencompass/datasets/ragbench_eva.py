from collections import Counter
from typing import List
import numpy as np
import re
import copy
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

# 这些正则表达式用于去除文本中的冠词（a、an、the）和标点符号。
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


# 将输入文本转换为小写，去除标点符号、冠词和多余的空格。
def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


class F1Metric:
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)  # 返回两个计数器中共同元素的最小计数
        num_same = sum(common.values())  # 计算预测值和参考答案中的共同项数
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute_each_pair(guess: str, answer: str):
        # 计算单个预测文本和参考答案的精确率、召回率和 F1 分数。它首先标准化文本，然后计算分数。
        if answer == "":
            return None, None, None
        if guess == "":
            return 0, 0, 0
        g_tokens = normalize_answer(guess).split()
        a_tokens = normalize_answer(answer).split()

        precision, recall, f1 = F1Metric._prec_recall_f1_score(g_tokens, a_tokens)
        return precision, recall, f1

    @staticmethod
    def compute_all_pairs(guesses: List[str], answers: List[list]):
        # 接受多个预测和对应的真实答案列表，对每一对进行F1得分的计算。
        # 如果一个答案中包含多个可能的正确回答，选择能得到最高F1得分的回答来计算。
        assert len(guesses) == len(answers)
        precision_list, recall_list, f1_list = [], [], []
        for guess, answer in zip(guesses, answers):
            assert type(answer) == list
            f1_list_tmp = []
            for answer_each in answer:
                answer_each = answer_each.strip()
                if answer_each == "":
                    continue
                precision, recall, f1 = F1Metric.compute_each_pair(guess, answer_each)
                f1_list_tmp.append(f1)

            if len(f1_list_tmp) > 0:
                f1 = max(f1_list_tmp)
                if precision is None or recall is None or f1 is None:
                    continue
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

        return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)


def compute_f1_score(predictions, references):
    guess_list = []
    for guess in predictions:
        guess = guess.strip()
        if "</s>" in guess:
            guess = guess.replace("</s>", "")
        guess_list.append(guess)

    answer_list = []
    for answer in references:
        answer_list.append(answer)

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    return f1


@ICL_EVALUATORS.register_module()
class RagBenchEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        f1 = compute_f1_score(predictions, references)
        return {'F1': f1 * 100}


@ICL_EVALUATORS.register_module()
class RagBench_inscit_Evaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        groundtruth_answers_update = []
        for answers in references:
            answers_update = []
            for ans in answers:
                ## this answer is additionally added to the answer_list for inscit dataset, needs to remove
                if ans != "Sorry. I cannot find the answer based on the context.":
                    answers_update.append(ans)
            assert len(answers_update) > 0
            groundtruth_answers_update.append(copy.deepcopy(answers_update))
        references = groundtruth_answers_update
        f1 = compute_f1_score(predictions, references)
        return {'F1': f1 * 100}


unanswerable_keyphrases = ["cannot find", "can't find", "not able to", "unable to", "does not provide",
                           "cannot provide", "cannot answer", "couldnot answer", "can't answer", "couldn't answer",
                           "cannot be found", "cannot be determined", "do not have", "couldn't find", "no information",
                           "does not mention", "doesn't mention", "not explicitly mentioned", "not explicitly explain",
                           "can not find", "could not find", "does not specify", "doesn't provide", "doesn't specify",
                           "there is no", "not mentioned", "don't have", "don't know", "does not include",
                           "doesn't include", "does not contain", "doesn't contain", "not provided",
                           "does not indicate", "doesn't indicate", "does not disclose", "doesn't disclose"]


@ICL_EVALUATORS.register_module()
class RagBench_quac_doqa_Evaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        predicted_answers_new = []
        for pred in predictions:
            pred = pred.lower()
            for keyphrase in unanswerable_keyphrases:
                if keyphrase in pred:
                    pred = "Sorry. I cannot find the answer based on the context."
                    break
            predicted_answers_new.append(pred)
        predictions = predicted_answers_new
        f1 = compute_f1_score(predictions, references)
        return {'F1': f1 * 100}


def _is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


@ICL_EVALUATORS.register_module()
class RagBench_convfinqa_Evaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        groundtruth_answers_formula = [item[0] for item in references]
        groundtruth_answers = [item[1] for item in references]
        question_list = [item[2] for item in references]
        count_exact_match = 0
        for question, pred, gold, gold_formula in zip(question_list, predictions, groundtruth_answers,
                                                      groundtruth_answers_formula):

            original_pred = pred
            ## convert 1,000,000 into 1000000
            original_pred = original_pred.replace(",", "")

            ## convert $10 million + $20 million into 10 + 20
            original_pred = original_pred.replace("$", "").replace("million", "").replace("billion", "")

            ## convert 10 (2017) + 20 (2018) into 10 + 20
            pattern = r'\((\b\w+\b)\)'
            original_pred = re.sub(pattern, '', original_pred)

            ## make sure it each token only has one space in between
            original_pred = " ".join(original_pred.split())

            if str(gold) in original_pred:
                count_exact_match += 1

            elif str(gold_formula) in original_pred:
                count_exact_match += 1

            elif _is_float(gold) and (
                    str(round(float(gold), 3)) in original_pred or str(round(float(gold), 2)) in original_pred):
                count_exact_match += 1

            elif "percent" in question and (
                    str(float(gold) * 100) in original_pred or str(round(float(gold) * 100, 1)) in original_pred or str(
                round(float(gold) * 100, 2)) in original_pred):
                count_exact_match += 1

            elif str(gold).endswith(".0") and str(int(gold)) in original_pred:
                ## gold is a integer like 80.0 then convert it into 80
                count_exact_match += 1

            elif "decrease" in original_pred and _is_float(gold) and gold < 0 and (str(-1 * gold) in original_pred):
                ## for the case where model generates something like a decrese of 10 million, while gold is -10.
                count_exact_match += 1

        match = (count_exact_match/len(predictions))
        return {'Accuracy': match * 100}
