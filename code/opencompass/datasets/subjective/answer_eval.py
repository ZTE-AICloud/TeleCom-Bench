import re
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseJudgeEvaluator:
    # def score(self, predictions, references, questions):
    #     return self._evaluate(predictions, references, questions, wiredops_judge_prompt, JudgeLlama())

    def _evaluate(self, predictions, references, questions, prompt, judge_model):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different lengths'
            }

        correct_count = 0
        valid_count = 0
        total_count = len(predictions)
        result_dict = {}

        def extract_judge(judge_message):
            if "1" in judge_message:
                return True
            elif "0" in judge_message:
                return False
            return None

        def process_prompt(question, prediction, reference):
            judge_prompt = prompt.format(question=question, prediction=prediction, reference=reference)
            print(judge_prompt)
            judge_message = judge_model.chat(judge_prompt)
            extract_judgment = extract_judge(judge_message)
            # 使用 question 的哈希值作为字典的 key
            key = hash(question)
            result_dict[key] = {
                'question': question,
                'prediction': prediction,
                'reference': reference,
                'is_correct': extract_judgment,
                'evaluation': judge_message,
            }
            return extract_judgment

        with ThreadPoolExecutor(max_workers=256) as executor:  # 设置线程数
            futures = [
                executor.submit(process_prompt, q, p, r)
                for q, p, r in zip(questions, predictions, references)
            ]

            for future in as_completed(futures):
                judgment = future.result()
                if judgment is not None:
                    valid_count += 1
                    if judgment is True:
                        correct_count += 1

        # 按 questions 的哈希顺序整理结果
        details = [result_dict[hash(q)] for q in questions]
        accuracy = float(correct_count / valid_count) if valid_count > 0 else 0
        return {
            'score': accuracy * 100,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'valid_count': valid_count,
            'total_count': total_count,
            'details': details
        }


class BaseScoreEvaluator:
    # def score(self, questions, predictions, references):
    #     return self._evaluate(questions, predictions, references, wiredops_score_prompt, JudgeLlama())

    def _evaluate(self, questions, predictions, references, prompt, judge_model):
        if len(predictions) != len(references):
            return {
                'error': 'prediction and reference have different length'
            }

        score_count = [0, 0, 0]
        valid_count = 0
        total_count = len(predictions)
        result_dict = {}

        def extract_score(judge_message):
            try:
                match = re.search(r'\s*分值：([0-9])', judge_message)
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
