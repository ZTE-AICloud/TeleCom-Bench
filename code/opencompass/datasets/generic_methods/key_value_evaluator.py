from collections import Counter
from typing import List

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import str2json


# kv严格一致
@ICL_EVALUATORS.register_module()
class KVStrictMatchEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []
        for pred, ref in zip(predictions, references):
            pred = str2json(pred.upper() if isinstance(pred, str) else "")
            ref = str2json(ref.upper() if isinstance(ref, str) else "")

            if pred is None or ref is None:
                correct = False
            else:
                # 必须所有key和value都完全一致
                correct = pred == ref

            is_correct.append(correct)
            processed_pred.append(pred)
            processed_gold.append(ref)

        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct,
            },
        }


@ICL_EVALUATORS.register_module()
class KVListMatchEvaluator(BaseEvaluator):
    """按指定 key_list 校验对应 KV 是否与标准答案一致（忽略大小写，空值等价）"""

    def __init__(self, key_list: List[str]):
        assert isinstance(key_list, list) and all(
            isinstance(k, str) for k in key_list
        ), "key_list 必须为字符串列表"
        # 统一大小写，后续比较忽略大小写
        self.key_list = [k.upper() for k in key_list]

    def _scalar_equal(self, a, b) -> bool:
        return a == b

    def _list_equal_ignore_order(self, a: list, b: list) -> bool:
        # 规则 1：长度>1 的列表，忽略顺序、保留重复元素
        try:
            return Counter(a) == Counter(b)
        except TypeError:
            try:
                return sorted(a, key=str) == sorted(b, key=str)
            except Exception:
                return False

    def _is_empty(self, v) -> bool:
        return (isinstance(v, list) and len(v) == 0) or (isinstance(v, str) and v == "")

    def _compare_by_rule(self, ref_v, pred_v) -> bool:
        # 空值等价：ref 为空([]或'')，pred 为[]或''均视为相等
        if self._is_empty(ref_v):
            return self._is_empty(pred_v)

        # 1) 标准答案为长度>1的列表：pred 必须为列表，忽略顺序比较
        if isinstance(ref_v, list) and len(ref_v) > 1:
            if not isinstance(pred_v, list):
                return False
            return self._list_equal_ignore_order(ref_v, pred_v)

        # 2) 标准答案为长度=1的列表：取 ref[0] 比较；pred 若为列表取 pred[0]，否则直接与 ref[0] 比较
        if isinstance(ref_v, list) and len(ref_v) == 1:
            ref_scalar = ref_v[0]
            # 单元素为空的等价判断：[] 或 ''
            if isinstance(ref_scalar, str) and ref_scalar == "":
                return self._is_empty(pred_v) or (
                    isinstance(pred_v, list)
                    and len(pred_v) > 0
                    and self._scalar_equal("", pred_v[0])
                )
            if isinstance(pred_v, list):
                if len(pred_v) == 0:
                    return False
                pred_scalar = pred_v[0]
            else:
                pred_scalar = pred_v
            return self._scalar_equal(ref_scalar, pred_scalar)

        # 3) 标准答案不是列表：pred 若为列表取 pred[0]，否则直接比较
        if isinstance(ref_v, str) and ref_v == "":
            return self._is_empty(pred_v)
        if isinstance(pred_v, list):
            if len(pred_v) == 0:
                return False
            pred_scalar = pred_v[0]
        else:
            pred_scalar = pred_v
        return self._scalar_equal(ref_v, pred_scalar)

    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        is_correct = []

        for pred, ref in zip(predictions, references):
            # 1) 将整段文本转为大写后再解析，达到键名与字符串值忽略大小写的效果
            pred_json = str2json(pred.upper() if isinstance(pred, str) else "")
            ref_json = str2json(ref.upper() if isinstance(ref, str) else "")

            processed_pred.append(pred_json)
            processed_gold.append(ref_json)

            if (
                pred_json is None
                or ref_json is None
                or not isinstance(ref_json, dict)
                or not isinstance(pred_json, dict)
            ):
                is_correct.append(False)
                continue

            # 2) key_list 中的键是否都在标准答案中
            missing_in_gold = [k for k in self.key_list if k not in ref_json]
            if missing_in_gold:
                is_correct.append(False)
                print(f"\nMissing key in gold: {missing_in_gold}\n")
                continue

            # 3) 继续判断模型回答的对应 KV 是否一致（模型缺键视为错误）
            missing_in_pred = [k for k in self.key_list if k not in pred_json]
            if missing_in_pred:
                is_correct.append(False)
                continue

            # 4) 指定键全部一样则正确（根据规则 1/2/3 + 空值等价 比较 value）
            equal = all(
                self._compare_by_rule(ref_json[k], pred_json[k]) for k in self.key_list
            )
            is_correct.append(equal)

        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0.0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "is_correct": is_correct,
            },
        }


@ICL_EVALUATORS.register_module()
class KVListMatchACCEvaluator(BaseEvaluator):
    """按指定 key_list 校验对应 KV 是否与标准答案一致，以每个key的准确率为分数"""

    def __init__(self, key_list: List[str]):
        assert isinstance(key_list, list) and all(
            isinstance(k, str) for k in key_list
        ), "key_list 必须为字符串列表"
        # 保存原始key列表用于返回结果
        self.original_key_list = key_list
        # 统一大小写，后续比较忽略大小写
        self.key_list = [k.upper() for k in key_list]

    def _scalar_equal(self, a, b) -> bool:
        return a == b

    def _list_equal_ignore_order(self, a: list, b: list) -> bool:
        # 规则 1：长度>1 的列表，忽略顺序、保留重复元素
        try:
            return Counter(a) == Counter(b)
        except TypeError:
            try:
                return sorted(a, key=str) == sorted(b, key=str)
            except Exception:
                return False

    def _is_empty(self, v) -> bool:
        return (isinstance(v, list) and len(v) == 0) or (isinstance(v, str) and v == "")

    def _compare_by_rule(self, ref_v, pred_v) -> bool:
        # 空值等价：ref 为空([]或'')，pred 为[]或''均视为相等
        if self._is_empty(ref_v):
            return self._is_empty(pred_v)

        # 1) 标准答案为长度>1的列表：pred 必须为列表，忽略顺序比较
        if isinstance(ref_v, list) and len(ref_v) > 1:
            if not isinstance(pred_v, list):
                return False
            return self._list_equal_ignore_order(ref_v, pred_v)

        # 2) 标准答案为长度=1的列表：取 ref[0] 比较；pred 若为列表取 pred[0]，否则直接与 ref[0] 比较
        if isinstance(ref_v, list) and len(ref_v) == 1:
            ref_scalar = ref_v[0]
            # 单元素为空的等价判断：[] 或 ''
            if isinstance(ref_scalar, str) and ref_scalar == "":
                return self._is_empty(pred_v) or (
                    isinstance(pred_v, list)
                    and len(pred_v) > 0
                    and self._scalar_equal("", pred_v[0])
                )
            if isinstance(pred_v, list):
                if len(pred_v) == 0:
                    return False
                pred_scalar = pred_v[0]
            else:
                pred_scalar = pred_v
            return self._scalar_equal(ref_scalar, pred_scalar)

        # 3) 标准答案不是列表：pred 若为列表取 pred[0]，否则直接比较
        if isinstance(ref_v, str) and ref_v == "":
            return self._is_empty(pred_v)
        if isinstance(pred_v, list):
            if len(pred_v) == 0:
                return False
            pred_scalar = pred_v[0]
        else:
            pred_scalar = pred_v
        return self._scalar_equal(ref_v, pred_scalar)

    def score(self, predictions: List[str], references: List[str]) -> dict:
        processed_pred = []
        processed_gold = []
        kv_detail = []  # 每个样本中每个key的判断结果

        for pred, ref in zip(predictions, references):
            # 1) 将整段文本转为大写后再解析，达到键名与字符串值忽略大小写的效果
            pred_json = str2json(pred.upper() if isinstance(pred, str) else "")
            ref_json = str2json(ref.upper() if isinstance(ref, str) else "")

            processed_pred.append(pred_json)
            processed_gold.append(ref_json)

            # 记录当前样本中每个key的判断结果（使用原始key名称）
            sample_kv_result = {}

            if (
                pred_json is None
                or ref_json is None
                or not isinstance(ref_json, dict)
                or not isinstance(pred_json, dict)
            ):
                # 如果解析失败，所有key都标记为False
                for original_k in self.original_key_list:
                    sample_kv_result[original_k] = False
                kv_detail.append(sample_kv_result)
                continue

            # 2) key_list 中的键是否都在标准答案中
            missing_in_gold = [k for k in self.key_list if k not in ref_json]
            if missing_in_gold:
                print(f"\nMissing key in gold: {missing_in_gold}\n")
                # 标准答案中缺失的key标记为False
                for original_k in self.original_key_list:
                    sample_kv_result[original_k] = False
                kv_detail.append(sample_kv_result)
                continue

            # 3) 对每个key单独判断是否匹配
            for original_k, upper_k in zip(self.original_key_list, self.key_list):
                # 如果预测答案中缺少该key，标记为False
                if upper_k not in pred_json:
                    sample_kv_result[original_k] = False
                else:
                    # 比较该key的value是否匹配
                    is_match = self._compare_by_rule(ref_json[upper_k], pred_json[upper_k])
                    sample_kv_result[original_k] = is_match

            kv_detail.append(sample_kv_result)

        # 计算每个key的准确率
        key_accuracies = {}
        num_samples = len(kv_detail)
        
        if num_samples == 0:
            avg_accuracy = 0.0
        else:
            # kv_detail中存储的是原始key名称
            for original_k in self.original_key_list:
                correct_count = sum(1 for sample_result in kv_detail if sample_result.get(original_k, False))
                key_acc = correct_count / num_samples
                key_accuracies[f"{original_k}_acc"] = key_acc * 100
            
            # 计算所有key的平均准确率
            avg_accuracy = sum(key_accuracies.values()) / len(key_accuracies) if key_accuracies else 0.0

        result = {
            "accuracy": avg_accuracy,
            "detail_dict": {
                "processed_pred": processed_pred,
                "processed_gold": processed_gold,
                "kv_detail": kv_detail,
            },
        }
        
        # 添加每个key的准确率
        result.update(key_accuracies)
        
        return result
