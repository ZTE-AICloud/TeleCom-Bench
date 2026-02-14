import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests


class Qwen235B:
    """OpenAI-compatible chat client with OpenRouter support."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    LOCAL_URL = "http://10.55.56.14:31005/v1/chat/completions"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "qwen/qwen3-235b-a22b-2507",
        retries: int = 3,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        proxy_url: Optional[str] = None,
        timeout: int = 15 * 60,
        temperature: float = 0,
        max_tokens: int = 8192,
    ):
        self.retries = max(1, retries)
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        if provider == "openrouter":
            self.base_url = base_url or self.OPENROUTER_URL
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OpenRouter requires api_key or OPENROUTER_API_KEY.")
            self.headers: Dict[str, str] = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            if site_url:
                self.headers["HTTP-Referer"] = site_url
            if app_name:
                self.headers["X-Title"] = app_name
        else:
            self.base_url = base_url or self.LOCAL_URL
            self.headers = {"Content-Type": "application/json"}

    @staticmethod
    def _normalize_messages(message: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if isinstance(message, list):
            return message
        return [{"content": message, "role": "user"}]

    @staticmethod
    def _extract_text_from_response(response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Invalid response, missing choices: {response_data}")
        return choices[-1]["message"]["content"]

    def _parse_stream_response(self, response: requests.Response) -> str:
        parts: List[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                parts.append(content)
            if choices[0].get("finish_reason"):
                break
        return "".join(parts)

    def chat(
        self,
        message: Union[str, List[Dict[str, str]]],
        stream: bool = False,
        detail: bool = False,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        messages = self._normalize_messages(message)

        payload = {
            "model": self.model,
            "stream": stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if kwargs:
            payload.update(kwargs)

        for attempt in range(self.retries):
            try:
                if stream:
                    with requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout,
                        stream=True,
                        proxies=self.proxies,
                    ) as response:
                        response.raise_for_status()
                        result_text = self._parse_stream_response(response)
                        return {"content": result_text} if detail else result_text

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                    proxies=self.proxies,
                )
                response.raise_for_status()
                response_data = response.json()
                if detail:
                    return response_data
                return self._extract_text_from_response(response_data)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return "LLM ERROR"

    def predict(self, input_text, **kwargs):
        return self.chat(message=input_text, **kwargs)

    def __call__(self, input_text, **kwargs):
        ans = self.predict(input_text, **kwargs)
        print(f"问题{input_text}\n模型回答:{ans}")
        return ans

def main() -> None:
    # Put all tunable model settings here for easy editing.
    provider = "openrouter"  # "openrouter" or "local"
    model = "qwen/qwen3-235b-a22b-2507"
    api_key = "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"
    base_url = None  # local example: "http://10.55.56.14:31005/v1/chat/completions"
    site_url = None  # optional, maps to HTTP-Referer
    app_name = "judge-service"  # optional, maps to X-Title
    proxy_url = "http://127.0.0.1:2220"  # company HTTP/HTTPS proxy
    retries = 3
    timeout = 15 * 60
    temperature = 0
    max_tokens = 8192

    client = Qwen235B(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        site_url=site_url,
        app_name=app_name,
        proxy_url=proxy_url,
        retries=retries,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = client.chat("请作为裁判模型，判断答案是否正确并简要说明理由。", stream=False)
    print(response)


if __name__ == "__main__":
    main()
