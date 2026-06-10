import mmengine


class BaseEvaluatorHandler:
    """评估器结果处理基类。

    子类可覆写:
      - get_extra_preds(test_set): 当 score() 参数名与 test_set 列名不一致时,
        返回 {pred_key: list_of_values} 进行补充注入。
      - process(...): 自定义打分与逐条结果回写逻辑。
    """

    @staticmethod
    def get_extra_preds(test_set):
        return {}

    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result = evaluator.score(**preds)
        if 'detail_dict' in result:
            detail_dict = result['detail_dict']
            for key, values in detail_dict.items():
                for i, value in enumerate(values):
                    origin_preds[str(i)][key] = value
            mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


# ---------------------------------------------------------------------------
# 具体 Handler 实现
# ---------------------------------------------------------------------------

class AccHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result, preprocessed = evaluator.score(**preds)
        is_corrects = [
            pred == ref for pred, ref in
            zip(preprocessed['predictions'], preprocessed['references'])
        ]
        if origin_preds:
            for idx, is_correct in enumerate(is_corrects):
                origin_preds[str(idx)]['is_correct'] = is_correct
        mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        return result


class HuggingfaceHandler(BaseEvaluatorHandler):
    @staticmethod
    def process(evaluator, preds, origin_preds, filename):
        result, _ = evaluator.score(**preds)
        return result


class CoreNetworkHandler(BaseEvaluatorHandler):
    @staticmethod
    def get_extra_preds(test_set):
        extra = {}
        if 'key_point' in test_set.features:
            extra['key_points'] = list(test_set['key_point'])
        if 'opt_point' in test_set.features:
            extra['opt_points'] = list(test_set['opt_point'])
        return extra


# ---------------------------------------------------------------------------
# 注册中心
# ---------------------------------------------------------------------------

class ResultsUpdate:
    """评估器结果处理注册中心。

    三级匹配顺序: 精确名称 -> 子串模式 -> 基类名称。
    新增评估器只需:
      1. 编写 Handler 子类
      2. 在 _handlers / _pattern_handlers / _base_class_handlers 中注册
      或运行时调用 ResultsUpdate.register_handler(...)
    """

    _handlers = {
        'AccEvaluator': AccHandler,
        'CoreNetworkEvaluator': CoreNetworkHandler,
    }

    _pattern_handlers = []

    _base_class_handlers = {
        'HuggingfaceEvaluator': HuggingfaceHandler,
    }

    @classmethod
    def get_handler(cls, evaluator):
        """根据评估器解析对应的 Handler。

        接受评估器实例或类名字符串。
        匹配顺序: 精确名称 -> 子串模式 -> 基类名称。
        """
        if isinstance(evaluator, str):
            name = evaluator
        else:
            name = evaluator.__class__.__name__

        if name in cls._handlers:
            return cls._handlers[name]

        for pattern, handler in cls._pattern_handlers:
            if pattern in name:
                return handler

        if not isinstance(evaluator, str):
            for base in evaluator.__class__.__mro__:
                if base.__name__ in cls._base_class_handlers:
                    return cls._base_class_handlers[base.__name__]

        return None

    @classmethod
    def register_handler(cls, evaluator_name, handler_class,
                         match_type='exact'):
        """运行时注册新 handler。

        Args:
            evaluator_name: 评估器类名或匹配模式。
            handler_class: BaseEvaluatorHandler 子类。
            match_type: 'exact' | 'pattern' | 'base_class'。
        """
        if match_type == 'exact':
            cls._handlers[evaluator_name] = handler_class
        elif match_type == 'pattern':
            cls._pattern_handlers.append((evaluator_name, handler_class))
        elif match_type == 'base_class':
            cls._base_class_handlers[evaluator_name] = handler_class
