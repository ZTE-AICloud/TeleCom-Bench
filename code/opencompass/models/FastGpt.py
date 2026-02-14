import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]

@MODELS.register_module()
class FastGpt(BaseAPIModel):
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
                 path: str = 'zteaim_api',
                 api_url: str = "",
                 api_data: Dict = {},
                 api_headers: Dict = {},
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 api_key = "",
                 retries = 3
                 ):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        self.api_key = api_key
        self.retries = retries
        self.base_url = api_url
        self.timeout =  15 * 60
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        start_time = time.time()
        #print(f"inputs 类型 {type(inputs)} {inputs}")
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        end_time = time.time()
        cost_time = end_time - start_time
        print(f"Batch 执行时间 {cost_time}  秒")
        return results

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        # print(f"input: {input}  {type(input)}")
        messages = []
        message = {}
        import uuid

        # 生成一个随机 UUID
        random_uuid = uuid.uuid4()
        chat_id = str(random_uuid)
        stream = True
        detail = False

        # for x in input:
        #     print(f"x : {x} xtype {type(x)}")

        if isinstance(input, PromptList):
            for x in input:
                #print(f"x : {x}")
                if x['role'] == "HUMAN":
                    message = {"role":"user", "content":x['prompt']}
                elif x['role'] == "BOT":
                    message = {"role":"assistant", "content":x['prompt']}
                elif x['role'] == "SYSTEM":
                    message = {"role":"system", "content":x['prompt']}
                messages.append(message)
        else:
            message = {'role':'user', 'content':input}
            messages.append(message)

        
        payload = {
            "chatId": chat_id,
            "stream": stream,
            "max_tokens":32000,
            "detail": detail,
            "messages": messages
        }
        for attempt in range(self.retries):
            try:
                if stream:
                    with requests.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout, stream=True) as response:
                        #response.raise_for_status()
                        content = r""
                        for line in response.iter_lines():
                            #print(content)
                            result = str(line, encoding='utf-8')
                            #print(result)
                            if "[DONE]" in result or len(result) == 0:
                                continue
                            elif '"finish_reason":"stop"' in result:
                                break
                            elif '"object":"error"' in result:
                                print(result)
                                break
                            else:
                                try:
                                    if 'content' in result:
                                        #print(result)
                                        result = re.search(r'"content":"(.*?)"', result).group(1)
                                        #result = result.replace("\\", "\n")
                                        #print(result)
                                        content += result
                                    else:
                                        content += ""
                                except Exception as e:
                                    print(e)
                                    continue
                        print(eval('f"' + content + '"'))
                        return  eval('f"' + content + '"')
                else:
                    response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout)
                    print(response.text)
                    response_data = response.json()
                    #print(response_data['choices'][-1])
                    content = response_data['choices'][-1]['message']['content']
                    return content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return "LLM ERROR"