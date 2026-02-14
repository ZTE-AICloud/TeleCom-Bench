import json
import os
import os.path as osp
from typing import List, Dict, Any

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.text_postprocessors import str2json

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class GaoJingDataset(BaseDataset):
    """告警根因分析数据集
    
    数据集结构：
    - 每个样本包含input.json（告警图数据）和label.json（标准根因答案）
    - 模型需要从告警图中推理出根因节点
    """

    @staticmethod
    def load(path: str) -> Dataset:
        """加载数据集
        
        Args:
            path: 数据集根目录路径，包含多个test_*子文件夹
        """
        data = []
        
        # 遍历所有test子文件夹
        test_folders = sorted([f for f in os.listdir(path) if f.startswith('test_')])
        
        for folder in test_folders:
            folder_path = osp.join(path, folder)
            input_file = osp.join(folder_path, "input.json")
            label_file = osp.join(folder_path, "label.json")            
            
            # 读取input和label文件
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 生成prompt
            prompt = _generate_prompt_template(input_data)
            
            # 提取标准答案（标签中的根因信息）
            label_root_causes = []
            for node in label_data.get('nodes', []):
                # 兼容两种label格式
                if node.get('label') == 'RootCause':
                    # 格式1: label字段标记类型
                    props = node.get('properties', {})
                    label_root_causes.append({
                        'cause_description': props.get('causeName', props.get('evalCause', '')),
                        'equipment_id': props.get('causeId', props.get('ldn', ''))
                    })
                elif 'RootCause' in node.get('labels', []):
                    # 格式2: labels数组标记类型
                    props = node.get('properties', {})
                    label_root_causes.append({
                        'cause_description': props.get('evalCause', props.get('causeName', '')),
                        'equipment_id': props.get('ldn', props.get('causeId', ''))
                    })
            
            data.append({
                'prompt': prompt,
                'label': label_root_causes,
                'case_id': folder
            })
        
        return Dataset.from_list(data)


def _generate_prompt_template(input_data: Dict[str, Any]) -> str:
    """生成告警根因分析的prompt模板"""
    prompt_template = f"""你需要对根因推理图<{input_data}>进行分析，分三个步骤输出：
第一步：识别图中所有告警节点，其中：
- 统计并列出 label 为 'TargetAlarm' 的节点数量和节点的 "title"；
- 统计并列出 @class 为 'AlarmDetail' 的节点数量和节点 id。
- 除了'TargetAlarm'外
第二步：对于每个 'TargetAlarm' 节点，从该节点出发，在图中寻找到所有可能的 'RootCause' 根因节点的路径：
- 所有路径必须为合法有向路径，遵循 'causeby' 或设备级联（如 'RiCable', 'dependOn'）关系；
- 路径可以跨越多个设备层级，需完整遍历；
- 每个跳转步骤都基于图中的实际边，不可跳跃、不猜测；
- 注意不同设备（即ldn）的根因，最终的结果不应该只推理出一个根因，做到不遗漏
第三步：根据每个路径，输出如下内容：
- 导致当前 TargetAlarm 的所有 RootCause 节点的 'title'字段作为根因名称；
- 每个根因对应的causeId，从该节点的 'ldn' 字段获取；
- 若 RootCause 节点中存在 'solution' 字段，则作为解决方案；若无则为空。
请将结果以如下 JSON 格式输出, alarmTitle仅为目标告警，不要加入其他告警：
{{
  "target_alarms": [
    {{
      "alarmTitle": "",
      "root_causes": [
        {{
          "cause_description": "",
          "causeId": "",
          "solution": ""
        }},
        {{
          "cause_description": "",
          "causeId": "",
          "solution": ""
        }}
      ]
    }}
  ]
}}
注意：非告警节点的边之间没有方向性关系，只表示连接关系，设备和告警之间的边表示的含义是告警是由设备上报的，告警与告警之间的边表示递进关系"""
    return prompt_template


@ICL_EVALUATORS.register_module()
class GaoJingEvaluator(BaseEvaluator):
    """告警根因分析评估器
    
    评估指标：
    - accuracy: 根因预测准确率（基于cause_description匹配）
    - f1: F1分数
    - recall: 召回率
    """

    def _extract_standard_json(self, text: str) -> dict:
        """从文本中提取标准JSON格式"""
        if not isinstance(text, str):
            return None
        
        try:
            # 移除思考标签
            if '</think>' in text:
                parts = text.split('</think>')
                if len(parts) > 1:
                    text = parts[-1]
                else:
                    return None
            
            # 查找JSON起止位置
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = text[start_idx:end_idx]
            
            # 使用系统提供的 str2json 解析
            json_obj = str2json(json_str)
            
            # 验证JSON结构
            if not isinstance(json_obj, dict) or 'target_alarms' not in json_obj:
                return None
            
            # 补全必需字段
            for alarm in json_obj.get('target_alarms', []):
                if 'alarmTitle' not in alarm:
                    alarm['alarmTitle'] = ""
                if 'root_causes' not in alarm:
                    alarm['root_causes'] = []
                
                for cause in alarm['root_causes']:
                    if 'cause_description' not in cause:
                        cause['cause_description'] = ""
                    if 'causeId' not in cause:
                        cause['causeId'] = ""
                    if 'solution' not in cause:
                        cause['solution'] = ""
            
            return json_obj
        
        except Exception:
            return None

    def _calculate_metrics(self, label_root_causes: List[Dict], pred_root_causes: List[Dict]) -> Dict[str, float]:
        """计算准确率、召回率和F1分数（基于cause_description）"""
        # 提取cause_description列表
        label_descriptions = [
            str(item.get('cause_description', '')).strip()
            for item in label_root_causes
            if isinstance(item, dict) and str(item.get('cause_description', '')).strip()
        ]
        
        pred_descriptions = [
            str(item.get('cause_description', '')).strip()
            for item in pred_root_causes
            if isinstance(item, dict) and str(item.get('cause_description', '')).strip()
        ]
        
        # 空值处理
        if not pred_descriptions and not label_descriptions:
            return {'accuracy': 1.0, 'f1': 1.0, 'recall': 1.0}
        
        if not pred_descriptions or not label_descriptions:
            return {'accuracy': 0.0, 'f1': 0.0, 'recall': 0.0}
        
        # 计算匹配
        label_set = set(label_descriptions)
        pred_set = set(pred_descriptions)
        
        tp = len(label_set & pred_set)
        fp = len(pred_set - label_set)
        fn = len(label_set - pred_set)
        
        # 计算指标
        accuracy = tp / len(pred_descriptions) if pred_descriptions else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall
        }

    def score(self, predictions: List[str], references: List[List[Dict]]) -> dict:
        """评分函数
        
        Args:
            predictions: 模型输出列表
            references: 标准答案列表（每个元素是根因字典列表）
        """
        accuracy_list = []
        f1_list = []
        recall_list = []
        
        for pred_text, label_root_causes in zip(predictions, references):
            # 提取模型输出的JSON
            pred_json = self._extract_standard_json(pred_text)
            
            if pred_json is None:
                # 解析失败，记录0分
                accuracy_list.append(0.0)
                f1_list.append(0.0)
                recall_list.append(0.0)
                continue
            
            
            # 提取预测的根因
            pred_root_causes = []
            for alarm in pred_json.get('target_alarms', []):
                for cause in alarm.get('root_causes', []):
                    pred_root_causes.append({
                        'cause_description': cause.get('cause_description', ''),
                        'equipment_id': cause.get('causeId', '')
                    })
            
            # 计算指标
            metrics = self._calculate_metrics(label_root_causes, pred_root_causes)
            accuracy_list.append(metrics['accuracy'])
            f1_list.append(metrics['f1'])
            recall_list.append(metrics['recall'])
        
        # 计算平均值
        n = len(accuracy_list)
        avg_accuracy = sum(accuracy_list) / n if n > 0 else 0.0
        avg_f1 = sum(f1_list) / n if n > 0 else 0.0
        avg_recall = sum(recall_list) / n if n > 0 else 0.0
        
        return {
            'F1': avg_f1 * 100,
            'acc': avg_accuracy * 100,
            'recall': avg_recall * 100,
        }

