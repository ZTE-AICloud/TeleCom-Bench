import re
import time
import threading
import logging
from typing import Optional, Dict, Any, List
from enum import Enum

import requests


class ErrorType(Enum):
    """错误类型枚举"""
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    AUTH_ERROR = "AUTH_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class TokenBucket:
    """令牌桶算法实现"""
    
    def __init__(self, capacity: int, fill_rate: float):
        """
        初始化令牌桶
        
        Args:
            capacity: 桶容量（最大令牌数）
            fill_rate: 填充速率（令牌/秒）
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        消费令牌
        
        Args:
            tokens: 要消费的令牌数量
            
        Returns:
            bool: 是否成功消费令牌
        """
        with self.lock:
            now = time.time()
            # 计算需要填充的令牌
            tokens_to_add = (now - self.last_update) * self.fill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_for_token(self, tokens: int = 1, max_wait: float = 60.0) -> bool:
        """
        等待令牌可用
        
        Args:
            tokens: 需要的令牌数量
            max_wait: 最大等待时间（秒）
            
        Returns:
            bool: 是否成功获得令牌
        """
        start_time = time.time()
        while not self.consume(tokens):
            if time.time() - start_time > max_wait:
                return False
            time.sleep(0.1)
        return True


class NTele72B:
    def __init__(self, base_url="http://10.5.212.158:7888/v1/chat/completions", 
                 model="NTele-72B-V3", retries=3, 
                 max_requests_per_second=3, max_tokens=4096):
        """
        初始化NTele72B模型客户端
        
        Args:
            base_url: API基础URL
            model: 模型名称
            retries: 重试次数
            max_requests_per_second: 每秒最大请求数
            max_tokens: 最大token数
        """
        self.retries = retries
        self.base_url = base_url
        self.model = model
        self.timeout = 15 * 60
        self.max_tokens = max_tokens
        self.temperature = 0
        
        # 令牌桶：每秒最多2个请求，桶容量为5
        self.rate_limiter = TokenBucket(capacity=10, fill_rate=max_requests_per_second)
        
        self.headers = {
            "Content-Type": "application/json",
            "authorization": "ZNHXTB-DJC-16baf5d0-662a-44f4-bacf-88e748625316"
        }
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _classify_error(self, error: Exception) -> ErrorType:
        """分类错误类型"""
        error_str = str(error).lower()
        
        if isinstance(error, requests.exceptions.Timeout):
            return ErrorType.TIMEOUT_ERROR
        elif isinstance(error, requests.exceptions.ConnectionError):
            return ErrorType.NETWORK_ERROR
        elif isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response is not None:
                status_code = error.response.status_code
                if status_code == 401:
                    return ErrorType.AUTH_ERROR
                elif status_code == 429:
                    return ErrorType.RATE_LIMIT_ERROR
                elif 500 <= status_code < 600:
                    return ErrorType.SERVER_ERROR
        elif "timeout" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "rate limit" in error_str or "too many requests" in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        
        return ErrorType.UNKNOWN_ERROR

    def _get_retry_delay(self, attempt: int, error_type: ErrorType) -> float:
        """根据错误类型和重试次数计算延迟时间"""
        base_delay = 1.0
        
        if error_type == ErrorType.RATE_LIMIT_ERROR:
            # 限流错误使用更长的延迟
            return base_delay * (2 ** attempt) + 5.0
        elif error_type == ErrorType.NETWORK_ERROR:
            # 网络错误使用指数退避
            return base_delay * (2 ** attempt)
        elif error_type == ErrorType.SERVER_ERROR:
            # 服务器错误使用较长的延迟
            return base_delay * (2 ** attempt) + 2.0
        else:
            # 其他错误使用标准指数退避
            return base_delay * (2 ** attempt)

    def _should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.retries:
            return False
        
        # 认证错误不重试
        if error_type == ErrorType.AUTH_ERROR:
            return False
        
        return True

    def chat(self, message, stream=False, detail=False):
        """
        发送聊天请求
        
        Args:
            message: 消息内容
            stream: 是否流式响应
            detail: 是否返回详细信息
            
        Returns:
            str: 响应内容或错误信息
        """
        if isinstance(message, list):
            messages = message
        else:
            messages = [
                {
                    "content": message,
                    "role": "user"
                }
            ]

        payload = {
            "model": self.model,
            "stream": stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        # 等待令牌可用
        if not self.rate_limiter.wait_for_token(max_wait=60.0):
            self.logger.error("Rate limit exceeded, unable to get token")
            return "RATE_LIMIT_ERROR"
        
        for attempt in range(self.retries + 1):
            try:
                self.logger.info(f"Sending request (attempt {attempt + 1}/{self.retries + 1})")
                
                if stream:
                    return self._handle_stream_response(payload)
                else:
                    return self._handle_normal_response(payload)
                    
            except Exception as e:
                error_type = self._classify_error(e)
                self.logger.warning(f"Attempt {attempt + 1} failed: {error_type.value} - {e}")
                
                if not self._should_retry(error_type, attempt):
                    self.logger.error(f"Not retrying due to error type: {error_type.value}")
                    return f"LLM_ERROR_{error_type.value}"
                
                # 计算重试延迟
                delay = self._get_retry_delay(attempt, error_type)
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        return "LLM_ERROR_MAX_RETRIES_EXCEEDED"

    def _handle_stream_response(self, payload: Dict[str, Any]) -> str:
        """处理流式响应"""
        with requests.post(self.base_url, headers=self.headers, json=payload, 
                          timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            content = ""
            
            for line in response.iter_lines():
                result = str(line, encoding='utf-8')
                
                if "[DONE]" in result or len(result) == 0:
                    continue
                elif '"finish_reason":"stop"' in result:
                    break
                elif '"object":"error"' in result:
                    self.logger.error(f"Stream error: {result}")
                    break
                else:
                    try:
                        if 'content' in result:
                            match = re.search(r'"content":"(.*?)"', result)
                            if match:
                                content += match.group(1)
                    except Exception as e:
                        self.logger.warning(f"Error parsing stream content: {e}")
                        continue
            
            return content

    def _handle_normal_response(self, payload: Dict[str, Any]) -> str:
        """处理普通响应"""
        response = requests.post(self.base_url, headers=self.headers, json=payload, 
                               timeout=self.timeout)
        response.raise_for_status()
        
        response_data = response.json()
        self.logger.debug(f"Response data: {response_data}")
        
        if 'choices' not in response_data or not response_data['choices']:
            raise ValueError("Invalid response format: no choices found")
        
        content = response_data['choices'][-1]['message']['content']
        return content.encode('utf-8').decode('utf-8')

    def predict(self, input_text, **kwargs):
        """预测接口"""
        return self.chat(message=input_text)

    def __call__(self, input_text, **kwargs):
        """调用接口"""
        ans = self.predict(input_text, **kwargs)
        self.logger.info(f"问题: {input_text}\n模型回答: {ans}")
        return ans

# 示例使用
# if __name__ == "__main__":
#     client = NTele72B()
#     response = client.chat(message="出一个4选1的选择题，并给出答案和解释", stream=False)
#     print(response)
