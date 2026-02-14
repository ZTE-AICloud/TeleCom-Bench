import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import httpx
from openai import OpenAI

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel

logger = logging.getLogger(__name__)

PromptType = Union[PromptList, str]


@MODELS.register_module()
class GeneralApi(BaseAPIModel):
    """Model wrapper for OpenAI-compatible API endpoints.

    This wrapper supports connection pooling, thread-safe operations,
    and automatic retry mechanisms for improved performance and reliability.

    Args:
        path (str): Model identifier. Defaults to 'zteaim_api'.
        api_url (str): Base URL of the API endpoint.
        api_data (Dict): Default request parameters (model, max_tokens, etc.).
        api_headers (Dict): HTTP headers including Authorization token.
            The Authorization header should contain the token without
            'Bearer ' prefix (it will be added automatically).
        query_per_second (int): Maximum queries per second. Defaults to 1.
        retry (int): Number of retries on API failure. Defaults to 2.
        meta_template (Dict, optional): Meta prompt template for message
            formatting.
        mode (str): Input truncation mode. Defaults to 'none'.
        stream (bool): Enable streaming responses. Defaults to True.
        top_p (float, optional): Top-p sampling parameter.
        enable_thinking (bool, optional): Enable thinking mode for reasoning models.
            When False, the parameter will be added to request as "enable_thinking": False.
            When True or None, the parameter will not be added (default behavior).
            Defaults to None.
    """

    is_api: bool = True

    DEFAULT_API_PARAMS = {"max_tokens": 2048, "temperature": 0.7, "stream": False}

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
            retry=retry,
        )
        self.headers = api_headers
        self.api_data = api_data
        self.api_url = api_url
        self.stream = stream
        self.timeout = 120

        auth_value = self.headers.get("Authorization", "")
        if not auth_value:
            logger.warning("Authorization 请求头为空，API 调用可能失败")
            self.api_key = ""
        elif auth_value.startswith("Bearer "):
            self.api_key = auth_value[7:]
            logger.debug("已从 Authorization 请求头中移除 'Bearer ' 前缀")
        else:
            self.api_key = auth_value

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

        self._init_http_client()
        self._optimize_system_params()
        self._client_lock = Lock()
        self._config_lock = Lock()

    def _init_http_client(self) -> None:
        """Initialize HTTP client with connection pooling."""
        try:
            limits = httpx.Limits(
                max_connections=100, max_keepalive_connections=50, keepalive_expiry=60.0
            )

            self.http_client = httpx.Client(
                limits=limits,
                timeout=httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=60.0),
                http2=True,
                transport=httpx.HTTPTransport(retries=1, http2=True),
            )

            default_headers = {
                k: v for k, v in self.headers.items() if k != "Authorization"
            }

            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=5 * 60,
                default_headers=default_headers,
                http_client=self.http_client,
            )
            logger.debug(f"HTTP 客户端初始化成功，base_url: {self.base_url}")
        except Exception as e:
            logger.error(f"初始化 HTTP 客户端失败: {e}", exc_info=True)
            raise

    def _optimize_system_params(self) -> None:
        """Optimize system parameters for better connection handling."""
        try:
            if os.geteuid() == 0:
                os.system("echo '1024 65535' > /proc/sys/net/ipv4/ip_local_port_range")
                os.system("echo 30 > /proc/sys/net/ipv4/tcp_fin_timeout")
                os.system("echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse")
                logger.info("系统参数优化成功")
            else:
                logger.debug("无 root 权限，跳过系统参数优化")
        except Exception as e:
            logger.warning(f"系统参数优化失败: {e}")

    def _exponential_backoff_retry(self, func, *args, **kwargs):
        """Retry with exponential backoff for connection errors."""
        max_retries = 3
        base_delay = 0.5

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException, socket.error) as e:
                if attempt == max_retries - 1:
                    logger.error(f"连接错误达到最大重试次数: {e}", exc_info=True)
                    raise
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"连接错误，{delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(delay)
            except Exception as e:
                logger.error(f"重试机制中的意外错误: {e}", exc_info=True)
                raise

    def generate(self, inputs: List[Union[str, PromptList]], **kwargs) -> List[str]:
        start_time = time.time()
        batch_size = len(inputs)
        logger.info(f"开始生成响应，batch_size: {batch_size}")

        max_workers = min(128, batch_size)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._generate, inputs))
            elapsed_time = time.time() - start_time
            logger.info(f"响应生成完成，耗时: {elapsed_time:.2f}秒")
            return results
        except Exception as e:
            logger.error(f"响应生成失败: {e}", exc_info=True)
            raise

    def _generate(self, input: Union[str, PromptList]) -> str:
        """Generate response for a single input."""
        messages = []

        try:
            if isinstance(input, PromptList):
                role_map = {"HUMAN": "user", "BOT": "assistant", "SYSTEM": "system"}
                for x in input:
                    if x["role"] in role_map:
                        # 跳过空的 SYSTEM 消息
                        if x["role"] == "SYSTEM" and not x.get("prompt", "").strip():
                            continue
                        messages.append(
                            {"role": role_map[x["role"]], "content": x["prompt"]}
                        )
            else:
                messages.append({"role": "user", "content": input})
        except (KeyError, TypeError) as e:
            logger.error(f"解析输入消息失败: {e}, 输入: {input}", exc_info=True)
            raise ValueError(f"无效的输入格式: {e}") from e

        MAX_RETRIES = 3
        retries = 0
        errors = []

        while retries < MAX_RETRIES:
            try:
                if self.stream:
                    result = self.generate_stream(messages)
                else:
                    result = self.generate_no_stream(messages)
                logger.debug(f"响应生成成功 (长度: {len(result) if result else 0})")
                return result
            except (httpx.ConnectError, httpx.TimeoutException, socket.error) as e:
                retries += 1
                errors.append(
                    f"尝试 {retries}: 连接错误 - {type(e).__name__}: {str(e)}"
                )

                if retries < MAX_RETRIES:
                    delay = 5 * (2 ** (retries - 1))
                    logger.warning(f"连接错误，{delay}秒后重试: {e}")
                    time.sleep(delay)
                else:
                    error_msg = f"达到最大重试次数。错误列表: {'; '.join(errors)}"
                    logger.error(error_msg, exc_info=True)
                    return f"错误: 超过最大重试次数。{error_msg}"
            except Exception as e:
                retries += 1
                errors.append(f"尝试 {retries}: {type(e).__name__}: {str(e)}")
                logger.error(f"生成过程中的意外错误: {e}", exc_info=True)

                if retries >= MAX_RETRIES:
                    error_msg = f"达到最大重试次数。错误列表: {'; '.join(errors)}"
                    logger.error(error_msg, exc_info=True)
                    return f"错误: 超过最大重试次数。{error_msg}"
                time.sleep(5)

        error_msg = f"生成失败，已重试 {MAX_RETRIES} 次。错误列表: {'; '.join(errors)}"
        logger.error(error_msg)
        return f"错误: {error_msg}"

    def _build_request_params(self, messages: List[Dict], stream: bool = False) -> Dict:
        """Build request parameters in a thread-safe manner."""
        with self._config_lock:
            params = self.DEFAULT_API_PARAMS.copy()
            params.update(self.api_data.copy())
            params["messages"] = messages
            params["stream"] = stream

            if self.enable_thinking is False:
                params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

            return params

    def generate_no_stream(self, messages: List[Dict]) -> str:
        """Generate non-streaming response."""
        try:
            request_params = self._build_request_params(messages, stream=False)

            with self._client_lock:
                completion = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create, **request_params
                )
                if not completion.choices or not completion.choices[0].message:
                    logger.error("API 返回空响应")
                    raise ValueError("API 返回空响应")
                return completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"非流式请求失败: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    def generate_stream(self, messages: List[Dict]) -> str:
        """Generate streaming response."""
        with self._client_lock:
            request_params = self._build_request_params(messages, stream=True)
            logger.debug(f"流式请求，消息数量: {len(messages)}")
            try:
                stream = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create, **request_params
                )
                chunks = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                result = "".join(chunks)
                logger.debug(f"流式响应完成 (长度: {len(result)})")
                return result
            except Exception as e:
                logger.error(
                    f"流式请求失败: {type(e).__name__}: {str(e)}", exc_info=True
                )
                raise

    def __del__(self):
        """Cleanup: close HTTP client on destruction."""
        try:
            if hasattr(self, "http_client"):
                self.http_client.close()
        except Exception as e:
            logger.debug(f"清理 HTTP 客户端时出错: {e}")
