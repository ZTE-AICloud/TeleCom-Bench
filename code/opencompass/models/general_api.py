import copy
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

from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel


class BaseGeneralApi(BaseAPIModel):
    is_api: bool = True

    DEFAULT_API_PARAMS = {
        'max_tokens': 2048,
        'temperature': 0.7,
        'stream': False
    }

    THINKING_CONTROL_MODE = 'none'  # none | prompt_suffix | extra_body
    PARSE_REASONING = True

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
        self.enable_thinking = enable_thinking

        auth_value = self.headers.get("Authorization", "")
        if not auth_value:
            self.logger.warning("Authorization 请求头为空，API 调用可能失败")
            self.api_key = ""
        elif auth_value.startswith("Bearer "):
            self.api_key = auth_value[7:]
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

        self._init_http_client()
        self._client_lock = Lock()

    def _init_http_client(self):
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=50,
            keepalive_expiry=60.0
        )

        self.http_client = httpx.Client(
            limits=limits,
            timeout=httpx.Timeout(
                connect=30.0,
                read=300.0,
                write=30.0,
                pool=60.0
            ),
            http2=True,
            transport=httpx.HTTPTransport(
                retries=1,
                http2=True
            )
        )

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

    def _exponential_backoff_retry(self, func, *args, **kwargs):
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException, socket.error) as e:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"连接错误，{delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
            except Exception as e:
                raise e

    def generate(
            self,
            inputs: List[Union[str, PromptList]],
            **kwargs
    ) -> List[Union[str, dict]]:
        start_time = time.time()
        batch_size = len(inputs)
        max_workers = min(64, batch_size)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._generate, inputs))
        end_time = time.time()
        self.logger.info(f"Batch 执行完成，batch_size: {batch_size}, 耗时: {end_time - start_time:.2f}秒")
        return results

    def _generate(self, input: List[Union[str, PromptList]]) -> Union[str, dict]:
        messages = []
        message = {}
        if isinstance(input, PromptList):
            for x in input:
                if x['role'] == "HUMAN":
                    message = {"role": "user", "content": x['prompt']}
                elif x['role'] == "BOT":
                    message = {"role": "assistant", "content": x['prompt']}
                elif x['role'] == "SYSTEM":
                    if x['prompt'] and x['prompt'].strip():
                        message = {"role": "system", "content": x['prompt']}
                        messages.append(message)
                    continue
                messages.append(message)
        else:
            messages.append({'role': 'user', 'content': input})

        retries = 0
        err_reason = ""
        while retries < 3:
            try:
                result = self.generate_stream(messages) if self.stream else self.generate_no_stream(messages)
                return result if isinstance(result, dict) else {'content': result}
            except (httpx.ConnectError, httpx.TimeoutException, socket.error) as e:
                retries += 1
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                err_reason += f"{timestamp} {retries}:Connection error - {type(e).__name__}: {str(e)}"
                if retries < 3:
                    delay = 5 * (2 ** (retries - 1))
                    self.logger.warning(f"连接错误，{delay}秒后重试: {e}")
                    time.sleep(delay)
                else:
                    self.logger.error("达到最大重试次数，放弃重试")
            except Exception as e:
                retries += 1
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                err_reason += f"{timestamp} {retries}:{type(e).__name__}: {str(e)}"
                self.logger.warning(f"请求失败，5秒后重试: {e}")
                time.sleep(5)

        return {'content': f"Max Retries, status: Error, err_reason:{err_reason}"}

    def _apply_prompt_suffix_control(self, messages: List[Dict]) -> List[Dict]:
        if self.enable_thinking is not False:
            return messages
        patched = copy.deepcopy(messages)
        for msg in reversed(patched):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str) and not content.rstrip().endswith('/no_think'):
                    msg['content'] = f"{content.rstrip()}\n/no_think"
                return patched
        if patched and isinstance(patched[-1].get('content', ''), str):
            content = patched[-1].get('content', '')
            if not content.rstrip().endswith('/no_think'):
                patched[-1]['content'] = f"{content.rstrip()}\n/no_think"
        return patched

    def _inject_extra_body_control(self, params: Dict) -> None:
        if self.enable_thinking is None:
            return
        extra_body = params.get('extra_body') or {}
        if not isinstance(extra_body, dict):
            extra_body = {}
        chat_template_kwargs = extra_body.get('chat_template_kwargs') or {}
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        chat_template_kwargs['enable_thinking'] = bool(self.enable_thinking)
        extra_body['chat_template_kwargs'] = chat_template_kwargs
        params['extra_body'] = extra_body

    def _prepare_messages_for_request(self, messages: List[Dict]) -> List[Dict]:
        if self.THINKING_CONTROL_MODE == 'prompt_suffix':
            return self._apply_prompt_suffix_control(messages)
        return messages

    def _build_request_params(self, messages, stream=False):
        request_messages = self._prepare_messages_for_request(messages)
        params = self.DEFAULT_API_PARAMS.copy()
        params.update(self.api_data)
        params['messages'] = request_messages
        params['stream'] = stream
        if self.THINKING_CONTROL_MODE == 'extra_body':
            self._inject_extra_body_control(params)
        return params

    def _extract_reasoning_from_message(self, message, result: Dict) -> None:
        if not self.PARSE_REASONING:
            return

        has_reasoning_field = False
        if hasattr(message, 'reasoning'):
            has_reasoning_field = True
            if message.reasoning:
                result['reasoning'] = message.reasoning

        if hasattr(message, 'model_dump'):
            try:
                message_dict = message.model_dump()
                if 'reasoning' in message_dict:
                    has_reasoning_field = True
                    if message_dict['reasoning']:
                        result['reasoning'] = message_dict['reasoning']
            except Exception:
                pass

        if isinstance(message, dict) and 'reasoning' in message:
            has_reasoning_field = True
            if message['reasoning']:
                result['reasoning'] = message['reasoning']

        if has_reasoning_field and 'reasoning' not in result:
            result['reasoning'] = None

    def _extract_message_field(self, message, field_name: str):
        has_field = False
        value = None

        if hasattr(message, field_name):
            has_field = True
            value = getattr(message, field_name)

        if hasattr(message, 'model_dump'):
            try:
                message_dict = message.model_dump()
                if field_name in message_dict:
                    has_field = True
                    value = message_dict[field_name]
            except Exception:
                pass

        if isinstance(message, dict) and field_name in message:
            has_field = True
            value = message[field_name]

        return has_field, value

    def generate_no_stream(self, messages):
        with self._client_lock:
            request_params = self._build_request_params(messages, stream=False)
            self.logger.info(f"完整请求参数: {json.dumps(request_params, indent=2, ensure_ascii=False)}")
            try:
                completion = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create,
                    **request_params
                )
                message = completion.choices[0].message
                has_content_field, ans = self._extract_message_field(message, 'content')
                has_reasoning_content_field, reasoning_content = self._extract_message_field(
                    message, 'reasoning_content')

                if (not has_content_field) or ans is None:
                    if has_reasoning_content_field:
                        ans = '' if reasoning_content is None else reasoning_content
                    else:
                        ans = None

                result = {'content': ans}
                if hasattr(message, 'refusal') and message.refusal is not None:
                    result['refusal'] = message.refusal
                if hasattr(message, 'model_dump'):
                    try:
                        message_dict = message.model_dump()
                        if 'refusal' in message_dict and message_dict['refusal'] is not None:
                            result['refusal'] = message_dict['refusal']
                    except Exception:
                        pass
                if isinstance(message, dict) and 'refusal' in message and message['refusal'] is not None:
                    result['refusal'] = message['refusal']

                self._extract_reasoning_from_message(message, result)

                self.logger.info(f"模型返回内容: {ans}")
                if 'reasoning' in result:
                    self.logger.info(f"模型返回 reasoning: {result['reasoning']}")
                return result
            except Exception as e:
                self.logger.error(f"非流式请求失败: {type(e).__name__}: {str(e)}, API URL: {self.api_url}")
                raise

    @staticmethod
    def _accumulate_text(current: Optional[str], piece: str) -> str:
        return piece if current is None else current + piece

    def _extract_reasoning_from_delta(self, delta):
        """Extract reasoning from a stream delta chunk.

        Returns (has_field: bool, value: Optional[str]).
        """
        if hasattr(delta, 'reasoning'):
            return True, delta.reasoning or None

        if hasattr(delta, 'model_dump'):
            try:
                delta_dict = delta.model_dump()
                if 'reasoning' in delta_dict:
                    return True, delta_dict['reasoning'] or None
            except Exception:
                pass

        return False, None

    def generate_stream(self, messages):
        with self._client_lock:
            request_params = self._build_request_params(messages, stream=True)
            self.logger.info(f"完整请求参数: {json.dumps(request_params, indent=2, ensure_ascii=False)}")
            try:
                stream = self._exponential_backoff_retry(
                    self.openai_client.chat.completions.create,
                    **request_params
                )
                text = None
                reasoning = None
                has_reasoning_field = False

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue

                    # Accumulate content
                    has_content, content_val = self._extract_message_field(delta, 'content')
                    has_rc, rc_val = self._extract_message_field(delta, 'reasoning_content')

                    if has_content and content_val is not None:
                        text = self._accumulate_text(text, content_val)
                    elif has_rc:
                        text = self._accumulate_text(text, '' if rc_val is None else rc_val)

                    # Accumulate reasoning
                    if self.PARSE_REASONING:
                        has_field, cur_val = self._extract_reasoning_from_delta(delta)
                        if has_field:
                            has_reasoning_field = True
                        if cur_val:
                            reasoning = cur_val if reasoning is None else reasoning + cur_val

                result = {'content': text}
                if self.PARSE_REASONING:
                    if reasoning:
                        result['reasoning'] = reasoning
                    elif has_reasoning_field:
                        result['reasoning'] = None

                self.logger.info(f"模型返回内容: {text}")
                if 'reasoning' in result:
                    self.logger.info(f"模型返回 reasoning: {result['reasoning']}")
                return result
            except Exception as e:
                self.logger.error(f"流式请求失败: {type(e).__name__}: {str(e)}, API URL: {self.api_url}")
                raise

    def __del__(self):
        try:
            if hasattr(self, 'http_client'):
                self.http_client.close()
        except Exception:
            pass
