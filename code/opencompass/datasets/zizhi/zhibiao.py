import json
import os
import os.path as osp
import re
from typing import List, Dict, Any, Optional

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


def _read_json_with_multiple_encodings(file_path: str) -> Any:
    """尝试使用多种编码读取JSON文件，与原评测脚本保持一致"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {file_path}")


@LOAD_DATASET.register_module()
class ZhiBiaoDataset(BaseDataset):
    """指标根因分析数据集
    
    数据集结构：
    - 每个样本包含input.json（指标数据）和label_converted.json（标准根因答案）
    - 模型需要从指标数据中推理出根因
    """

    @staticmethod
    def load(path: str, use_rule_kg: bool = True) -> Dataset:
        """加载数据集
        
        Args:
            path: 数据集根目录路径，包含多个InnerHO_test_*子文件夹
            use_rule_kg: 是否使用静态规则图谱
        """
        data = []
        rule_kg = None
        
        # 加载静态规则图谱，使用多种编码尝试
        if use_rule_kg:
            rule_kg_path = osp.join(path, "InnerHO_KG_cleaned.json")
            if osp.exists(rule_kg_path):
                rule_kg = _read_json_with_multiple_encodings(rule_kg_path)
        
        # 遍历所有test子文件夹
        test_folders = sorted([f for f in os.listdir(path) if f.startswith('InnerHO_test_')])
        
        for folder in test_folders:
            folder_path = osp.join(path, folder)
            input_file = osp.join(folder_path, "input.json")
            label_file = osp.join(folder_path, "label_converted.json")
            
            # 使用多种编码尝试读取文件，与原评测脚本保持一致
            input_data = _read_json_with_multiple_encodings(input_file)
            label_data = _read_json_with_multiple_encodings(label_file)
            
            # 提取标准答案
            label_root_causes = [
                {'cause_description': item['analysis_result']}
                for item in label_data if 'analysis_result' in item
            ]
            
            data.append({
                'prompt': _generate_prompt_template(input_data, rule_kg),
                'label': label_root_causes,
                'case_id': folder
            })
        
        return Dataset.from_list(data)


def _generate_prompt_template(input_data: Dict[str, Any], rule_kg: Any = None) -> str:
    """生成指标根因分析的prompt模板，与原评测脚本的prompt_generator完全一致"""
    if rule_kg is None:
        prompt_template = f"""
    你需要对如下数据<{input_data}>进行分析，目标是得出根因结论：
请将结果以如下 JSON 格式和样例输出（多个或者一个）：
{{
    "root_causes": [
        {{
          "cause_description": "根因码1:根因1",
        }},
        {{
          "cause_description": "根因码2:根因2",
        }}
      ]
}}
"""
    else:
        prompt_template = f"""
    你需要对如下原始数据<{input_data}>进行分析，并结合静态规则图谱<{rule_kg}>，目标是得出根因结论：
请将结果以如下 JSON 格式和样例输出（多个或者一个）：
{{
    "root_causes": [
        {{
          "cause_description": "M6010601002:系统内切换入执行失败，由于目标侧参数错配",
        }},
        {{
          "cause_description": "M6010601012:系统内切换出执行失败，由于小区负荷高",
        }}
      ]
}}
这些根因 "M6010601012:系统内切换出执行失败，由于小区负荷高" 位于静态规则图谱的"CheckItem"的"description"中，你需要分析原始数据，并从中归纳出Condition条件，再基于Condition在CheckItem中找到对应的根因
"""
    return prompt_template


@ICL_EVALUATORS.register_module()
class ZhiBiaoEvaluator(BaseEvaluator):
    """指标根因分析评估器
    
    评估指标：
    - accuracy: 根因预测准确率
    - precision: 精确率
    - recall: 召回率
    - f1: F1分数
    
    注意：评测前会清除文本中的标点符号
    """

    def _clean_punctuation(self, text: str) -> str:
        """清除文本中的标点符号"""
        if isinstance(text, str):
            # 移除所有标点符号，只保留字母、数字、中文字符和空格
            return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text).strip()
        return text

    def _clean_root_causes_list(self, root_causes_list: List[Dict]) -> List[Dict]:
        """清除根因列表中所有文本字段的标点符号"""
        return [
            {k: self._clean_punctuation(v) if isinstance(v, str) else v for k, v in rc.items()}
            for rc in root_causes_list
        ]

    def _calculate_accuracy(self, label_root_causes: List[Dict], pred_root_causes: List[Dict]) -> float:
        """计算准确率，与原脚本的calculate_accuracy逻辑完全一致"""
        cleaned_label = self._clean_root_causes_list(label_root_causes)
        cleaned_pred = self._clean_root_causes_list(pred_root_causes)

        # 处理空标签的特殊情况
        if len(cleaned_label) == 0:
            return 1.0 if len(cleaned_pred) == 0 else 0.0

        # 构建标签字典，key为cause_description
        label_dict = {item['cause_description']: item for item in cleaned_label}
        
        # 初始化计数器（使用预测数量作为分母，与原脚本一致）
        total_label = len(cleaned_pred)  # 注意：这里使用预测数量作为分母
        total_correct = 0

        # 遍历预测结果（与原脚本逻辑一致）
        for pred in cleaned_pred:
            if not isinstance(pred, dict):
                continue
            pred_id = pred.get('cause_description', '')
            # 只处理存在于标签中的预测
            if pred_id in label_dict:
                label_item = label_dict[pred_id]
                # 检查原因描述是否完全匹配（与原脚本第198行逻辑一致）
                is_cause_correct = (pred.get('cause_description', '') == label_item.get('cause_description', ''))
                # 检查整个元素是否完全匹配（与原脚本第209行逻辑一致）
                # 由于只有cause_description字段，所以完全匹配就是cause_description匹配
                if is_cause_correct:
                    total_correct += 1

        # 计算准确率（以预测数量为分母，与原脚本第214行一致）
        return total_correct / total_label if total_label > 0 else 0.0
    
    def _calculate_f1_score(self, label_root_causes: List[Dict], pred_root_causes: List[Dict]) -> Dict[str, float]:
        """计算精确率、召回率和F1分数，与原脚本的calculate_f1_score逻辑完全一致"""
        cleaned_label = self._clean_root_causes_list(label_root_causes)
        cleaned_pred = self._clean_root_causes_list(pred_root_causes)

        n_label = len(cleaned_label)
        n_pred = len(cleaned_pred)

        # 构建标签和预测的映射字典（与原脚本第232-238行一致）
        label_dict = {item['cause_description']: item for item in cleaned_label}
        pred_dict = {}
        for item in cleaned_pred:
            cause_desc = item.get('cause_description', '')
            if cause_desc not in pred_dict:
                pred_dict[cause_desc] = []
            pred_dict[cause_desc].append(item)

        # 初始化计数器（与原脚本第241-247行一致）
        tp_element = 0  # 整体元素真正例
        attr_names = ['cause_description']
        tp_attrs = {attr: 0 for attr in attr_names}  # 各属性真正例
        fn_attrs = {attr: 0 for attr in attr_names}  # 各属性假反例

        # 1. 计算整体元素指标和召回率相关指标（与原脚本第249-284行一致）
        for label_id, label_item in label_dict.items():
            # 检查该标签元素是否被完全匹配
            fully_matched = False
            # 检查各属性是否匹配
            attr_matched = {attr: False for attr in attr_names}

            if label_id in pred_dict:
                for pred_item in pred_dict[label_id]:
                    # 检查是否完全匹配
                    is_fully_matched = True
                    for attr in attr_names:
                        if pred_item.get(attr, '') != label_item.get(attr, ''):
                            is_fully_matched = False
                            break

                    # 更新完全匹配状态
                    if is_fully_matched:
                        fully_matched = True

                    # 更新各属性匹配状态
                    for attr in attr_names:
                        if not attr_matched[attr] and pred_item.get(attr, '') == label_item.get(attr, ''):
                            attr_matched[attr] = True

            # 更新整体元素计数器
            if fully_matched:
                tp_element += 1

            # 更新各属性计数器（召回率分母）
            for attr in attr_names:
                if attr_matched[attr]:
                    tp_attrs[attr] += 1
                else:
                    fn_attrs[attr] += 1

        # 2. 计算精确率相关指标（与原脚本第286-313行一致）
        fp_element = 0  # 整体元素假正例
        fp_attrs = {attr: 0 for attr in attr_names}  # 各属性假正例

        for pred in cleaned_pred:
            pred_id = pred.get('cause_description', '')
            # 检查是否完全匹配
            is_fully_matched = False
            if pred_id in label_dict:
                label_item = label_dict[pred_id]
                is_fully_matched = True
                for attr in attr_names:
                    if pred.get(attr, '') != label_item.get(attr, ''):
                        is_fully_matched = False
                        break

            # 更新整体元素计数器
            if not is_fully_matched:
                fp_element += 1

            # 更新各属性计数器
            for attr in attr_names:
                if pred_id in label_dict:
                    label_item = label_dict[pred_id]
                    if pred.get(attr, '') != label_item.get(attr, ''):
                        fp_attrs[attr] += 1
                else:  # cause_description不在标签中
                    fp_attrs[attr] += 1

        # 3. 计算整体元素指标（与原脚本第315-318行一致）
        precision_element = tp_element / n_pred if n_pred > 0 else 0
        recall_element = tp_element / n_label if n_label > 0 else 0
        f1_element = 2 * (precision_element * recall_element) / (precision_element + recall_element) if (precision_element + recall_element) > 0 else 0

        # 返回overall指标（与原脚本第322-326行一致）
        return {'precision': precision_element, 'recall': recall_element, 'f1': f1_element}

    def _extract_root_causes_from_pred_text(self, pred_text: str) -> Optional[List[Dict]]:
        """从模型原始输出中提取 root_causes 列表，与原脚本的extract_standard_json逻辑完全一致"""
        if not isinstance(pred_text, str) or not pred_text.strip():
            return None

        try:
            text = pred_text.strip()
            
            # 查找 JSON 开始和结束的位置（与原脚本第91-92行一致）
            start_idx = text.find('{')
            if start_idx == -1:
                return None
            
            end_idx = text.rfind('}') + 1
            
            # 提取JSON字符串（与原脚本第98-102行一致）
            if end_idx == 0:
                json_str = text[start_idx:]
            else:
                json_str = text[start_idx:end_idx]
            
            # 尝试解析JSON，如果失败则尝试修复（与原脚本第105-134行一致）
            try:
                json_obj = json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试修复不完整的JSON（缺少闭合括号）
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                open_brackets = json_str.count('[')
                close_brackets = json_str.count(']')
                
                missing_braces = open_braces - close_braces
                missing_brackets = open_brackets - close_brackets
                
                if missing_braces > 0 or missing_brackets > 0:
                    json_str_fixed = json_str.rstrip()
                    # 先闭合数组
                    if missing_brackets > 0:
                        json_str_fixed += '\n    ]'
                    # 再闭合对象
                    if missing_braces > 0:
                        json_str_fixed += '\n}'
                    try:
                        json_obj = json.loads(json_str_fixed)
                    except json.JSONDecodeError:
                        return None
                else:
                    return None
            
            # 验证JSON结构（与原脚本第136-146行一致）
            if not isinstance(json_obj, dict):
                return None
            
            if 'root_causes' not in json_obj:
                return None
            
            # 确保所有必需的字段都存在
            for cause in json_obj['root_causes']:
                if not all(key in cause for key in ['cause_description']):
                    return None
            
            return json_obj['root_causes']

        except Exception:
            return None

    def _calculate_average(self, values: List[float]) -> float:
        """计算平均值并转换为百分制"""
        return sum(values) / len(values) * 100 if values else 0.0

    def score(self, predictions: List[str], references: List[List[Dict]]) -> dict:
        """评分函数
        
        Args:
            predictions: 模型原始输出文本列表
            references: 标准答案列表（每个元素是根因字典列表）
        
        Returns:
            dict: 包含总体指标（百分制）和每道题详细信息的字典
        """
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        # 每道题的详细信息列表
        detail_dict = {
            "processed_pred": [],
            "processed_gold": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        
        for pred_text, label_root_causes in zip(predictions, references):
            if not isinstance(label_root_causes, list):
                raise TypeError(
                    f"references 的元素必须是 List[Dict]，但实际类型为 {type(label_root_causes)}"
                )

            # 提取预测的根因列表
            pred_root_causes = self._extract_root_causes_from_pred_text(pred_text)
            
            # 记录处理后的预测和标准答案
            detail_dict["processed_pred"].append(pred_root_causes if pred_root_causes else [])
            detail_dict["processed_gold"].append(label_root_causes)
            
            if pred_root_causes is None:
                # 解析失败，所有指标为0（与原脚本一致，解析失败时指标为0）
                detail_dict["accuracy"].append(0.0)
                detail_dict["precision"].append(0.0)
                detail_dict["recall"].append(0.0)
                detail_dict["f1"].append(0.0)
                continue

            # 计算各项指标
            accuracy = self._calculate_accuracy(label_root_causes, pred_root_causes)
            f1_metrics = self._calculate_f1_score(label_root_causes, pred_root_causes)

            detail_dict["accuracy"].append(accuracy)
            detail_dict["precision"].append(f1_metrics['precision'])
            detail_dict["recall"].append(f1_metrics['recall'])
            detail_dict["f1"].append(f1_metrics['f1'])
        
        # 计算总体指标（百分制）
        accuracy = self._calculate_average(detail_dict["accuracy"])
        precision = self._calculate_average(detail_dict["precision"])
        recall = self._calculate_average(detail_dict["recall"])
        f1 = self._calculate_average(detail_dict["f1"])
        
        return {
            "F1": f1,
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            # "average": (accuracy + precision + recall + f1) / 4,
            "detail_dict": detail_dict,
        }

