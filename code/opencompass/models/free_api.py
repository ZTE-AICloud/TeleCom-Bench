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
OPENAI_API_BASE = 'http://10.230.218.211:8004/generate'

@MODELS.register_module()
class FreeAPI(BaseAPIModel):
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

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 tokenizer_path = "",
                 max_tokens = 8096,
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 mode: str = 'none',
                 temperature: Optional[float] = None):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        self.max_tokens = max_tokens
        self.tokenizer_path = tokenizer_path
        if tokenizer_path != "":
            self.tokenizer_path = tokenizer_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if isinstance(key, str):
            self.keys = [os.getenv('OPENAI_API_KEY') if key == 'ENV' else key]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = openai_api_base
        self.path = path
        
        print(f'openai_api_base: {openai_api_base}')
        print(f'self.url: {self.url}')

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
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

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = input
        else:
            # print("=======================input is PromptList========================")
            # self.logger.info("=======================input is PromptList========================")
            # self.logger.info(f"PromptList: {PromptList}")
            if len(input) == 1:
                messages = input[0]['prompt']
            else:
                messages = ""
                for item in input:
                    msg = item['prompt']
                    messages = messages + msg

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0

                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]
            headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
            proxies = {'http':''}
            try:
                payload = dict(
                    prompt=messages,
                    max_tokens=self.max_tokens,
                    stop=['<|im_end|>', '<|endoftext|>', '</s>'],
                    temperature=temperature,
                )
                data = json.dumps(payload)
                self.logger.info(f'===============request payload: {payload}')
                # self.logger.info(f"=============data: {data}")
                raw_response = requests.post(self.url,
                                                headers=headers,
                                                data=data,
                                                verify=False,
                                                proxies=proxies
                                                )
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                    str(raw_response.content))
                # continue
                break
            try:
                response_text = response['text'][0].strip()
                if messages in response_text:
                    trim_resp = response_text.replace(messages, "")
                else:
                    trim_resp = response_text[len(messages)-1:]
                self.logger.info(f"response.content: {response_text}")
                self.logger.info(f"after trim: {trim_resp}")
                return trim_resp
            except:
                self.logger.error(f'decode response error, response:{str(response)}')
            max_num_retries += 1

        raise RuntimeError('Calling FastChat API failed after retrying for '
                        f'{max_num_retries} times. Check the logs for '
                        'details.')
