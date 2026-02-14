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
class ZteApiV1NoThink(BaseAPIModel):
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
        'max_tokens': 512,
        'temperature': 0.7,
        'stream': False
    }

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
        self.api_key = self.headers.get("Authorization", "Fake_Authorization")
        self.base_url = api_url.split("v1")[0] + "v1"

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
        self.openai_client = OpenAI(
            api_key=f"Bearer {self.api_key}",
            base_url=self.base_url,
            timeout=5 * 60,
            default_headers=self.headers,
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

                print("系统参数优化完成")
            else:
                print("无root权限，跳过系统参数优化")
        except Exception as e:
            print(f"系统参数优化失败: {e}")

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
                print(f"连接错误，{delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
            except Exception as e:
                # 其他错误直接抛出
                raise e

    def generate(
            self,
            inputs: List[Union[str, PromptList]],
            **kwargs
    ) -> List[str]:
        start_time = time.time()
        batch_size = len(inputs)
        print(f"Batch_size:{batch_size}")

        # 降低并发数，避免端口耗尽
        max_workers = min(64, batch_size)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(self._generate, inputs)
            )
        end_time = time.time()
        cost_time = end_time - start_time
        print(f"Batch 执行时间 {cost_time}  秒")
        return results

    def _generate(self, input: List[Union[str, PromptList]]) -> str:
        messages = []
        message = {}
        if isinstance(input, PromptList):
            for x in input:
                if x['role'] == "HUMAN":
                    # 在用户消息末尾添加 /no_think
                    content = x['prompt'] + "/no_think"
                    message = {"role": "user", "content": content}
                elif x['role'] == "BOT":
                    message = {"role": "assistant", "content": x['prompt']}
                elif x['role'] == "SYSTEM":
                    message = {"role": "system", "content": x['prompt']}
                messages.append(message)
        else:
            # 在普通输入消息末尾添加 /no_think
            content = input + "/no_think"
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
                print(f"模型回答:{result}")
                return result
            except (httpx.ConnectError, httpx.TimeoutException, socket.error) as e:
                # 明确捕获连接相关错误
                retries += 1
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                err_reason += f"{timestamp} {retries}:Connection error - {type(e).__name__}: {str(e)}"

                if retries < MAX_RETRIES:
                    # 指数退避重试
                    delay = 5 * (2 ** (retries - 1))  # 5s, 10s, 20s
                    print(f"连接错误，{delay}秒后重试: {e}")
                    time.sleep(delay)
                else:
                    print(f"达到最大重试次数，放弃重试")
            except Exception as e:
                # 其他错误
                retries += 1
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                err_reason += f"{timestamp} {retries}:{type(e).__name__}: {str(e)}"
                time.sleep(5)

        result = f"Max Retries, status: Error, err_reason:{err_reason}"
        return result

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
            try:
                completion = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create,
                    **request_params
                )
                ans = completion.choices[0].message.content
                return ans
            except Exception as e:
                print(f"非流式请求失败: {type(e).__name__}: {str(e)}")
                print(f"API URL: {self.api_url}")
                raise

    def generate_stream(self, messages):
        """流式请求，使用共享的HTTP客户端"""
        with self._client_lock:
            print("请求消息")
            request_params = self._build_request_params(messages, stream=True)
            try:
                stream = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create,
                    **request_params
                )
                text = ""
                for chunk in stream:
                    text += chunk.choices[0].delta.content or ""
                return text
            except Exception as e:
                print(f"流式请求失败: {type(e).__name__}: {str(e)}")
                print(f"API URL: {self.api_url}")
                raise

    def __del__(self):
        """析构函数，确保HTTP客户端正确关闭"""
        try:
            if hasattr(self, 'http_client'):
                self.http_client.close()
        except Exception:
            pass
