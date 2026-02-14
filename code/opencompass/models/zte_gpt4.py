# -*- coding: utf-8 -*-
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from transformers import AutoTokenizer

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]
OPENAI_API_BASE = 'https://rdcloud.zte.com.cn/zte-studio-ai-platform/openapi/v1/chat'


@MODELS.register_module()
class ZteGPT4(BaseAPIModel):
    def __init__(self,
                 path: str = 'gpt-4-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 key: Union[str, List[str]] = '80366441f8a9480eb312943a0a411054-23baa365c70b49a5afc7520bc57ae299',
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 empNo: str='',
                 authValue: str='',
                 mode: str = 'none',
                 temperature: Optional[float] = None):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        import tiktoken
        tiktoken_cache_dir = "/home/apollo/opencompass/tokenizer/"
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
        assert os.path.exists(os.path.join(tiktoken_cache_dir, '9b5ad71b2ce5302211f9c61530b329a4922fc6a4')) 
        self.tiktoken = tiktoken
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        self.key = key
        self.empNo = empNo
        self.authValue = authValue

        self.url = openai_api_base
        self.path = path
        print(f'ZteGPT4 init is called, self.url: {self.url}')

        if isinstance(key, str):
            print("key is str, value is", key)
            self.key = os.getenv('OPENAI_API_KEY') if key == 'ENV' else key
            print(self.key)
        else:
            print("key is not str, value is ", key)
            self.key = key
            print("self.key is ", self.key)

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 4096,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: str or PromptList, max_out_len: int, temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,text
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        # max num token for gpt-3.5-turbo is 4097
        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'bot'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)
        
        for item in input:
            if isinstance(item, dict) and 'prompt' in item:
                # 适配可能的输入错误，将 'prompt' 改为 'content'
                item['content'] = item.pop('prompt')
            messages.append(item)

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        max_out_len = min(
            max_out_len, context_window - self.get_token_len(str(input)) - 100)
        if max_out_len <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            header = {
                'Authorization': f'Bearer {self.key}',
                'X-Emp-No': self.empNo,
                'X-Auth-Value': self.authValue,
                'content-type': 'application/json',
            }

            try:
                data = dict(
                    chatUuid="",
                    chatName="222",
                    stream=False,
                    keep=False,
                    messages=input,
                    model=self.path,
                )

                # print("Headers:", header)
                # print("Data:", json.dumps(data))
                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))
                # print("Response Status Code:", raw_response.status_code)
                # print("Response Body:", raw_response.msg)
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
                if response["code"]["msg"] != "Success":
                    errorinfo = response["code"]
                    print("failed, rsp is ", response)
                    return ""
            except requests.JSONDecodeError:
                print('JsonDecode error, got: ', str(raw_response.content))
                return ""
            try:
                return response['bo']['result'].strip()
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        print("rate_limit_exceeded")
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        print(f'insufficient_quota key: {self.key}')
                        time.sleep(1)
                        continue
                    self.logger.error('Find error message in response: ', str(response['error']))
                else:
                    print("KeyError")
                    print(response)
            max_num_retries += 1

        print(f'Calling FastChat API failed after retrying for {max_num_retries} times. Check the logs for details.')
        return ""

    def get_token_len(self, prompt: str) -> int:
        model_name = self.path
        if self.path == "gpt-35-tuibo":
            model_name = "gpt-3.5-turbo"
        enc = self.tiktoken.encoding_for_model(model_name)
        return len(enc.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Get a suffix of prompt which is no longer than num_token tokens.

        Args:
            prompt (str): Input string.
            num_token (int): The upper bound of token numbers.

        Returns:
            str: The trimmed prompt.
        """
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt
