import mmengine


class BaseEvaluatorHandler:
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        raise NotImplementedError("Subclasses must implement this method")


class NebulabizAbstractHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        metrics, cut_results, length_results = evaluator.score(**preds)
        for idx, (cut_result, length_result) in enumerate(zip(cut_results, length_results)):
            origin_preds[str(idx)]['is_pred_intact'] = cut_result
            origin_preds[str(idx)]['is_valid_length'] = length_result
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return metrics


class NebulabizRepeatTruncatedHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        metrics, repeat_results, truncate_results = evaluator.score(**preds)
        for idx, (repeat_result, truncate_result) in enumerate(zip(repeat_results, truncate_results)):
            origin_preds[str(idx)]['repeat_eval'] = repeat_result
            origin_preds[str(idx)]['truncated_eval'] = truncate_results[idx]
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return metrics


class NebulabizFieldsQAHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        judge_message_list, point_list = result['judge_message_list'], result['point_list']
        for idx, (judge_message, point) in enumerate(zip(judge_message_list, point_list)):
            origin_preds[str(idx)]['judge_message'] = judge_message
            origin_preds[str(idx)]['point'] = point
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class NebulabizAbstractFormatHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        point_list = result['point_list']
        for idx, point in enumerate(point_list):
            origin_preds[str(idx)]['point'] = point
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class NebulabizValidJsonHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        is_valid_json_list = result['is_valid_json_list']
        for idx, is_valid_json in enumerate(is_valid_json_list):
            origin_preds[str(idx)]['is_valid_json'] = is_valid_json
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class NebulabizTranslateHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        details = result['details']
        if_pass_list = [item['if_pass'] for item in details]
        remark_list = [item['remark'] for item in details]

        for idx, (if_pass, remark) in enumerate(zip(if_pass_list, remark_list)):
            origin_preds[str(idx)]['if_pass'] = if_pass
            origin_preds[str(idx)]['remark'] = remark

        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class RanJSONKeyValueHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        preds = result['preds']
        refs = result['refs']
        correct_list = result['correct_list']
        for idx, (pred, ref, is_correct) in enumerate(zip(preds, refs, correct_list)):
            origin_preds[str(idx)]['processed_pred'] = pred
            origin_preds[str(idx)]['processed_gold'] = ref
            origin_preds[str(idx)]['is_correct'] = is_correct
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class RanTableSelectHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        preds = result['preds']
        refs = result['refs']
        correct_list = result['correct_list']
        for idx, (pred, ref, is_correct) in enumerate(zip(preds, refs, correct_list)):
            origin_preds[str(idx)]['processed_pred'] = pred
            origin_preds[str(idx)]['processed_gold'] = ref
            origin_preds[str(idx)]['is_correct'] = is_correct
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class WiredChoiceJudgeHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        judge_list = result['judge_list']
        for idx, judge_res in enumerate(judge_list):
            origin_preds[str(idx)]['judge_res'] = judge_res
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class ResultsUpdate:
    handlers = {
        'NebulabizAbstractEvaluator': NebulabizAbstractHandler,
        'NebulabizRepeatTruncateEvaluator': NebulabizRepeatTruncatedHandler,
        "NebulabizFieldsQA10Evaluator": NebulabizFieldsQAHandler,
        "NebulabizFieldsQA3Evaluator": NebulabizFieldsQAHandler,
        "NebulabizAbstractFormatEvaluator": NebulabizAbstractFormatHandler,
        "NebulabizValidJsonEvaluator": NebulabizValidJsonHandler,
        "NebulabizTranslateEvaluator": NebulabizTranslateHandler,
        "RanJSONKeyValueEvaluator": RanJSONKeyValueHandler,
        "RanTableSelectEvaluator": RanTableSelectHandler,
        "WiredChoiceJudgeEvaluator": WiredChoiceJudgeHandler

    }

    @classmethod
    def get_handler(cls, evaluator_name):
        handler_class = cls.handlers.get(evaluator_name)
        if not handler_class:
            raise ValueError(f"No handler found for evaluator: {evaluator_name}")
        return handler_class
