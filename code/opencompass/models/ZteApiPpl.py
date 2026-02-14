import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union, Tuple

import jieba
import requests
import torch
import numpy as np
from openai import OpenAI

import traceback

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from opencompass.utils.logging import get_logger

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class ZteApiPpl(BaseAPIModel):
    """Model wrapper around ZTE API for PPL evaluation.

    Args:
        path (str): The name of the model.
        api_url (str): The URL of the API endpoint.
        api_data (Dict): The data to be sent to the API.
        api_headers (Dict): The headers to be sent with the API request.
        max_seq_len (int): The maximum allowed sequence length of a model.
        query_per_second (int): The maximum queries allowed per second.
        retry (int): Number of retries if the API call fails.
        meta_template (Dict, optional): The model's meta prompt template.
        mode (str): The method of input truncation.
        stream (bool): Whether to use streaming response.
        ppl_config (Dict, optional): Configuration for PPL calculation.
    """

    is_api: bool = True

    def __init__(self,
                 path: str = 'zteaim_api',
                 api_url: str = "",
                 api_data: Dict = {},
                 api_headers: Dict = {},
                 max_seq_len: int = 1024,
                 query_per_second: int = 1,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 stream: bool = True,
                 ppl_config: Optional[Dict] = None
                 ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        self.headers = api_headers
        self.api_data = api_data
        self.api_url = api_url
        self.stream = stream
        self.timeout = 120
        self.model = api_data['model']
        self.api_key = self.headers.get("Authorization", "Fake_Authorization")
        self.base_url = api_url.split("v1")[0] + "v1"
        self.logger = get_logger()
        
        # PPL相关配置
        self.ppl_config = ppl_config or {}
        self.token_cache = {}  # 用于缓存tokenize结果
        self.logits_cache = {}  # 用于缓存logits结果

    def _tokenize(self, text: Union[str, PromptList]) -> List[int]:
        """对文本进行tokenize
        
        Args:
            text (Union[str, PromptList]): 输入文本或PromptList
            
        Returns:
            List[int]: token id列表
        """
        # 如果是PromptList，将其转换为字符串
        if isinstance(text, PromptList):
            text = self.parse_template(text, mode='ppl')
            
        # 如果是列表，将其转换为字符串
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
            
        # 检查缓存
        if text in self.token_cache:
            return self.token_cache[text]
            
        messages = [{"role": "user", "content": f"请对以下文本进行tokenize，返回token id列表：{text}"}]
        data = self.api_data.copy()
        data['messages'] = messages
        data['task'] = 'tokenize'
        
        try:
            self.logger.info(f"Sending tokenize request with data: {json.dumps(data, ensure_ascii=False)}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=self.timeout
            )
            self.logger.info(f"Tokenize response status code: {response.status_code}")
            self.logger.info(f"Tokenize response content: {response.text}")
            
            if response.status_code == 200:
                response_json = response.json()
                self.logger.info(f"Tokenize response JSON: {json.dumps(response_json, ensure_ascii=False)}")
                if 'tokens' not in response_json:
                    self.logger.error(f"Response JSON missing 'tokens' key: {response_json}")
                    raise KeyError("'tokens' key not found in response")
                tokens = response_json['tokens']
                # 更新缓存
                self.token_cache[text] = tokens
                return tokens
            else:
                raise Exception(f"Tokenize failed with status code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Tokenize error: {e}")
            self.logger.error(f"Tokenize error traceback: {traceback.format_exc()}")
            raise

    def _get_logits(self, tokens: List[int]) -> torch.Tensor:
        """获取token序列的logits
        
        Args:
            tokens (List[int]): token id列表
            
        Returns:
            torch.Tensor: logits张量
        """
        # 检查缓存
        token_key = str(tokens)
        if token_key in self.logits_cache:
            return self.logits_cache[token_key]
            
        messages = [{"role": "user", "content": f"请计算以下token序列的logits：{tokens}"}]
        data = self.api_data.copy()
        data['messages'] = messages
        data['task'] = 'get_logits'
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=self.timeout
            )
            if response.status_code == 200:
                logits = torch.tensor(response.json()['logits'])
                # 更新缓存
                self.logits_cache[token_key] = logits
                return logits
            else:
                raise Exception(f"Get logits failed with status code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Get logits error: {e}")
            raise

    def get_ppl(self,
                inputs: List[Union[str, PromptList]],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """计算困惑度分数
        
        Args:
            inputs (List[Union[str, PromptList]]): 输入文本列表
            mask_length (Optional[List[int]]): mask长度列表
            
        Returns:
            List[float]: 困惑度分数列表
        """
        ppl_scores = []
        
        for i, input_text in enumerate(inputs):
            try:
                # 1. 对输入进行tokenize
                tokens = self._tokenize(input_text)
                
                # 2. 获取logits
                logits = self._get_logits(tokens)
                
                # 3. 计算困惑度
                loss = 0
                valid_tokens = 0
                
                # 处理mask_length
                start_idx = 0
                if mask_length is not None and i < len(mask_length):
                    start_idx = mask_length[i]
                
                for j in range(start_idx, len(tokens)-1):
                    # 获取当前token的预测概率
                    probs = torch.softmax(logits[j], dim=-1)
                    # 获取下一个token的id
                    next_token_id = tokens[j+1]
                    # 计算交叉熵
                    loss -= torch.log(probs[next_token_id])
                    valid_tokens += 1
                
                if valid_tokens > 0:
                    # 计算平均损失
                    avg_loss = loss / valid_tokens
                    # 计算困惑度
                    ppl = torch.exp(avg_loss).item()
                else:
                    ppl = float('inf')
                    
                ppl_scores.append(ppl)
                
            except Exception as e:
                self.logger.error(f"Error calculating PPL for input {i}: {e}")
                ppl_scores.append(float('inf'))
                
        return ppl_scores

    def get_ppl_from_template(self,
                             templates: Union[PromptType, List[PromptType]],
                             mask_length=None) -> torch.Tensor:
        """从模板获取困惑度
        
        Args:
            templates (Union[PromptType, List[PromptType]]): 模板或模板列表
            mask_length (List[int], optional): mask长度列表
            
        Returns:
            torch.Tensor: 困惑度分数张量
        """
        inputs = self.parse_template(templates, mode='ppl')
        # 确保inputs是列表
        if not isinstance(inputs, list):
            inputs = [inputs]
        ppl_scores = self.get_ppl(inputs, mask_length)
        # 将列表转换为torch.tensor
        return torch.tensor(ppl_scores, dtype=torch.float32)

    def get_logits(self, inputs: List[Union[str, PromptList]]) -> Tuple[torch.Tensor, Dict]:
        """获取模型的logits输出
        
        Args:
            inputs (List[Union[str, PromptList]]): 输入文本列表
            
        Returns:
            Tuple[torch.Tensor, Dict]: logits张量和输入信息
        """
        all_logits = []
        all_tokens = []
        
        for input_text in inputs:
            try:
                tokens = self._tokenize(input_text)
                logits = self._get_logits(tokens)
                all_logits.append(logits)
                all_tokens.append(tokens)
            except Exception as e:
                self.logger.error(f"Error getting logits for input: {e}")
                raise
                
        return torch.stack(all_logits), {'tokens': all_tokens}

    def parse_template(self, prompt_template: Union[PromptType, List[PromptType]], mode: str) -> Union[str, List[str]]:
        """解析模板
        
        Args:
            prompt_template (Union[PromptType, List[PromptType]]): 模板或模板列表
            mode (str): 模式，'ppl'或'gen'
            
        Returns:
            Union[str, List[str]]: 解析后的文本或文本列表
        """
        # 如果是列表，递归处理每个元素
        if isinstance(prompt_template, list):
            results = [self.parse_template(item, mode) for item in prompt_template]
            # 如果所有结果都是字符串，则返回字符串
            if all(isinstance(r, str) for r in results):
                return ' '.join(results)
            return results
        
        # 如果是字符串，直接返回
        if isinstance(prompt_template, str):
            return prompt_template
            
        # 如果是字典，尝试提取prompt字段
        if isinstance(prompt_template, dict):
            if 'prompt' in prompt_template:
                return prompt_template['prompt']
            elif 'content' in prompt_template:
                return prompt_template['content']
            else:
                # 如果没有找到prompt或content字段，尝试将所有值连接起来
                return ' '.join(str(v) for v in prompt_template.values())
        
        # 如果是PromptList，转换为字符串
        if isinstance(prompt_template, PromptList):
            text = ""
            for item in prompt_template:
                if item['role'] in ['HUMAN', 'BOT', 'system']:
                    text += item['prompt'] + "\n"
            return text.strip()
        
        # 如果是元组，转换为列表处理
        if isinstance(prompt_template, tuple):
            return self.parse_template(list(prompt_template), mode)
        
        # 如果都不匹配，尝试使用父类的方法
        try:
            return super().parse_template(prompt_template, mode)
        except Exception as e:
            self.logger.error(f"Failed to parse template: {e}")
            self.logger.error(f"Template type: {type(prompt_template)}")
            self.logger.error(f"Template content: {prompt_template}")
            raise ValueError(f"Unsupported prompt template type: {type(prompt_template)}")

    def clear_cache(self):
        """清除缓存"""
        self.token_cache.clear()
        self.logits_cache.clear()

    def generate(self, inputs: List[str], max_out_len: int = 512, **kwargs) -> List[str]:
        """保持原有的生成功能"""
        # 复用ZteApiV1的generate方法
        return super().generate(inputs, max_out_len, **kwargs)
    