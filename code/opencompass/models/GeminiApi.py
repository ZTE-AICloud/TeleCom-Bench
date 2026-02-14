import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Union

import httpx
from openai import OpenAI

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class CustomApi(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
    """

    is_api: bool = True

    # 默认请求参数
    DEFAULT_API_PARAMS = {
        'max_tokens': 2048,
        'temperature': 0.7,
        'stream': False
    }

    def __init__(
        self,
        path: str = "zteaim_api",
        api_url: str = "",
        api_data: Dict = {},
        api_headers: Dict = {},
        query_per_second: int = 1,
        retry: int = 2,
        meta_template: Optional[Dict] = None,
        mode: str = "none",
        stream: bool = True,
        top_p: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
    ):
        super().__init__(
            path=path,
            meta_template=meta_template,
            query_per_second=query_per_second,
            retry=retry
        )
        self.headers = api_headers
        self.api_data = api_data
        self.api_url = api_url
        self.stream = stream
        self.timeout = 120

        # 处理 api_key
        auth_value = self.headers.get("Authorization", "")
        if not auth_value:
            self.logger.warning("Authorization 请求头为空，API 调用可能失败")
            self.api_key = ""
        elif auth_value.startswith("Bearer "):
            self.api_key = auth_value[7:]
        else:
            self.api_key = auth_value

        # 处理 base_url，与 GeneralApi 一致
        if "/v1" in api_url:
            v1_index = api_url.rfind("/v1")
            self.base_url = api_url[:v1_index] + "/v1"
        elif api_url.endswith("/chat/completions"):
            self.base_url = api_url[:-len("/chat/completions")]
        else:
            self.base_url = api_url

        if top_p is not None:
            self.DEFAULT_API_PARAMS["top_p"] = top_p

        self.enable_thinking = enable_thinking

        # 初始化HTTP客户端和连接池
        self._init_http_client()

        # 系统参数优化
        self._optimize_system_params()

        # 线程锁，用于客户端复用
        self._client_lock = Lock()

    def _init_http_client(self):
        """初始化HTTP客户端，配置连接池"""
        # 创建连接池配置
        limits = httpx.Limits(
            max_connections=100,  # 最大连接数
            max_keepalive_connections=50,  # 保持连接数
            keepalive_expiry=60.0  # 连接保持时间（秒）
        )

        # 创建HTTP客户端
        self.http_client = httpx.Client(
            limits=limits,
            timeout=httpx.Timeout(
                connect=30.0,  # 连接超时
                read=300.0,  # 读取超时（5分钟）
                write=30.0,  # 写入超时
                pool=60.0  # 连接池超时
            ),
            # 启用连接复用
            http2=True,
            # 配置重试
            transport=httpx.HTTPTransport(
                retries=1,  # 传输层重试次数
                http2=True
            )
        )

        # 创建OpenAI客户端，使用共享的HTTP客户端
        # 处理 default_headers，移除 Authorization 避免重复
        default_headers = {
            k: v for k, v in self.headers.items() if k != "Authorization"
        }
        
        self.openai_client = OpenAI(
            api_key=f"Bearer {self.api_key}",
            base_url=self.base_url,
            timeout=5 * 60,
            default_headers=default_headers,
            http_client=self.http_client
        )

    def _optimize_system_params(self):
        """优化系统参数，扩大端口范围，加速连接回收"""
        try:
            # 尝试扩大端口范围（需要root权限）
            if os.geteuid() == 0:  # 检查是否有root权限
                # 扩大端口范围到1024-65535
                os.system("echo '1024 65535' > /proc/sys/net/ipv4/ip_local_port_range")

                # 减少TIME_WAIT时间，加速连接回收
                os.system("echo 30 > /proc/sys/net/ipv4/tcp_fin_timeout")
                os.system("echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse")
                os.system("echo 1 > /proc/sys/net/ipv4/tcp_tw_recycle")

                self.logger.debug("系统参数优化完成")
        except Exception as e:
            self.logger.debug(f"系统参数优化失败: {e}")

    def _exponential_backoff_retry(self, func, *args, **kwargs):
        """指数退避重试机制"""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException, socket.error) as e:
                if attempt == max_retries - 1:
                    raise e

                # 指数退避：1s, 2s, 4s
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"连接错误，{delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
            except Exception as e:
                # 其他错误直接抛出
                raise e

    def generate(
            self,
            inputs: List[Union[str, PromptList]],
            **kwargs
    ) -> List[Union[str, dict]]:
        start_time = time.time()
        batch_size = len(inputs)

        # 降低并发数，避免端口耗尽
        max_workers = min(64, batch_size)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(self._generate, inputs)
            )
        end_time = time.time()
        cost_time = end_time - start_time
        self.logger.info(f"Batch 执行完成，batch_size: {batch_size}, 耗时: {cost_time:.2f}秒")
        return results

    def _generate(self, input: List[Union[str, PromptList]]) -> Union[str, dict]:
        messages = []
        message = {}
        if isinstance(input, PromptList):
            for x in input:
                if x['role'] == "HUMAN":
                    # 只在 enable_thinking=False 时，在用户消息末尾添加 /no_think
                    content = x['prompt'] + " /no_think" if self.enable_thinking is False else x['prompt']
                    message = {"role": "user", "content": content}
                elif x['role'] == "BOT":
                    message = {"role": "assistant", "content": x['prompt']}
                elif x['role'] == "SYSTEM":
                    # 如果 SYSTEM 消息内容为空，则跳过
                    if x['prompt'] and x['prompt'].strip():
                        message = {"role": "system", "content": x['prompt']}
                        messages.append(message)
                    continue
                messages.append(message)
        else:
            # 只在 enable_thinking=False 时，在普通输入消息末尾添加 /no_think
            content = input + "/no_think" if self.enable_thinking is False else input
            message = {'role': 'user', 'content': content}
            messages.append(message)

        MAX_RETRIES = 3
        retries = 0
        err_reason = ""

        while retries < MAX_RETRIES:
            try:
                if self.stream:
                    result = self.generate_stream(messages)
                else:
                    result = self.generate_no_stream(messages)
                # 如果返回的是字典，直接返回；如果是字符串，包装成字典以保持兼容性
                if isinstance(result, dict):
                    return result
                else:
                    # 向后兼容：如果返回的是字符串，包装成字典
                    return {'content': result}
            except (httpx.ConnectError, httpx.TimeoutException, socket.error) as e:
                # 明确捕获连接相关错误
                retries += 1
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                err_reason += f"{timestamp} {retries}:Connection error - {type(e).__name__}: {str(e)}"

                if retries < MAX_RETRIES:
                    # 指数退避重试
                    delay = 5 * (2 ** (retries - 1))  # 5s, 10s, 10s
                    self.logger.warning(f"连接错误，{delay}秒后重试: {e}")
                    time.sleep(delay)
                else:
                    self.logger.error("达到最大重试次数，放弃重试")
            except Exception as e:
                # 其他错误
                retries += 1
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                err_reason += f"{timestamp} {retries}:{type(e).__name__}: {str(e)}"
                self.logger.warning(f"请求失败，{5}秒后重试: {e}")
                time.sleep(5)

        error_result = f"Max Retries, status: Error, err_reason:{err_reason}"
        return {'content': error_result}

    def _build_request_params(self, messages, stream=False):
        params = self.DEFAULT_API_PARAMS.copy()
        params.update(self.api_data)
        params['messages'] = messages
        params['stream'] = stream  # 强制覆盖，确保调用时传入的stream生效
        return params

    def generate_no_stream(self, messages):
        """非流式请求，使用共享的HTTP客户端"""
        with self._client_lock:
            request_params = self._build_request_params(messages, stream=False)
            # 打印完整请求参数（不截断）
            self.logger.info(f"完整请求参数: {json.dumps(request_params, indent=2, ensure_ascii=False)}")
            
            try:
                completion = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create,
                    **request_params
                )
                message = completion.choices[0].message
                ans = message.content
                
                # 构建返回结果字典，包含 content 和可能的额外字段
                result = {'content': ans}
                
                # 尝试提取可能的额外字段（如 reasoning, refusal 等）
                # 方法1: 通过属性访问
                if hasattr(message, 'reasoning') and message.reasoning:
                    result['reasoning'] = message.reasoning
                if hasattr(message, 'refusal') and message.refusal is not None:
                    result['refusal'] = message.refusal
                
                # 方法2: 通过 model_dump() 转换为字典后访问（适用于 Pydantic 模型）
                try:
                    if hasattr(message, 'model_dump'):
                        message_dict = message.model_dump()
                        if 'reasoning' in message_dict and message_dict['reasoning']:
                            result['reasoning'] = message_dict['reasoning']
                        if 'refusal' in message_dict and message_dict['refusal'] is not None:
                            result['refusal'] = message_dict['refusal']
                except Exception:
                    pass
                
                # 方法3: 如果 message 本身就是字典类型
                if isinstance(message, dict):
                    if 'reasoning' in message and message['reasoning']:
                        result['reasoning'] = message['reasoning']
                    if 'refusal' in message and message['refusal'] is not None:
                        result['refusal'] = message['refusal']
                
                # 记录模型返回的内容
                self.logger.info(f"模型返回内容: {ans}")
                if 'reasoning' in result:
                    self.logger.info(f"模型返回 reasoning: {result['reasoning']}")
                
                return result
            except Exception as e:
                self.logger.error(f"非流式请求失败: {type(e).__name__}: {str(e)}, API URL: {self.api_url}")
                raise

    def generate_stream(self, messages):
        """流式请求，使用共享的HTTP客户端"""
        with self._client_lock:
            request_params = self._build_request_params(messages, stream=True)
            # 打印完整请求参数（不截断）
            self.logger.info(f"完整请求参数: {json.dumps(request_params, indent=2, ensure_ascii=False)}")
            
            try:
                stream = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create,
                    **request_params
                )
                text = ""
                reasoning = None
                refusal = None
                
                debug_chunk_count = 0
                for chunk in stream:
                    # --- DEBUG: 打印前 5 个 chunk 的完整内容，用于定位字段 ---
                    if debug_chunk_count < 5:
                        try:
                            # 尝试多种方式获取原始数据
                            chunk_data = None
                            if hasattr(chunk, 'model_dump'):
                                chunk_data = chunk.model_dump()
                            elif hasattr(chunk, 'to_dict'):
                                chunk_data = chunk.to_dict()
                            elif hasattr(chunk, '__dict__'):
                                chunk_data = chunk.__dict__
                            
                            self.logger.info(f"DEBUG RAW CHUNK {debug_chunk_count}: {json.dumps(chunk_data, default=str, ensure_ascii=False) if chunk_data else chunk}")
                        except Exception as e:
                             self.logger.info(f"DEBUG RAW CHUNK {debug_chunk_count} log error: {e}")
                        debug_chunk_count += 1
                    # ----------------------------------------------------

                    # 检查 choices 是否存在且非空
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta:
                            if hasattr(delta, 'content') and delta.content:
                                text += delta.content
                            
                            # 尝试提取 reasoning 字段
                            # 某些模型（如 Gemini/DeepSeek）可能在 OpenAI 兼容接口中返回标准 SDK 未定义的字段
                            current_reasoning = None
                            
                            # 1. 优先尝试直接属性访问
                            if hasattr(delta, 'reasoning') and delta.reasoning:
                                current_reasoning = delta.reasoning
                            
                            # 2. 如果属性没有，尝试通过 model_dump (Pydantic V2) 获取额外字段
                            # 注意：避免与属性访问重复获取
                            elif hasattr(delta, 'model_dump'):
                                try:
                                    # model_dump() 通常包含所有字段，包括未定义的额外字段
                                    delta_dict = delta.model_dump()
                                    if 'reasoning' in delta_dict and delta_dict['reasoning']:
                                        current_reasoning = delta_dict['reasoning']
                                except Exception:
                                    pass
                            
                            if current_reasoning:
                                reasoning = current_reasoning if reasoning is None else reasoning + current_reasoning
                
                # 构建返回结果字典
                result = {'content': text}
                if reasoning:
                    result['reasoning'] = reasoning
                
                # 记录模型返回的内容
                self.logger.info(f"模型返回内容: {text}")
                if reasoning:
                    self.logger.info(f"模型返回 reasoning: {reasoning}")
                
                return result
            except Exception as e:
                self.logger.error(f"流式请求失败: {type(e).__name__}: {str(e)}, API URL: {self.api_url}")
                raise

    def __del__(self):
        """析构函数，确保HTTP客户端正确关闭"""
        try:
            if hasattr(self, 'http_client'):
                self.http_client.close()
        except Exception:
            pass
