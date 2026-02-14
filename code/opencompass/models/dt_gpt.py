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
class DtGPT(BaseAPIModel):
    def __init__(self,
                 path: str = 'gpt-35-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 ):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        self.url = "https://rdcloud.zte.com.cn/zte-studio-ai-platform/openapi/v1/chat"
        self.model = path
        self.headers = {
            'Authorization': 'Bearer 80366441f8a9480eb312943a0a411054-23baa365c70b49a5afc7520bc57ae299',
            'Content-Type': 'application/json',
            'X-Emp-No': '00054055',
            'X-Auth-Value': '5a856d5b6fc3b0a431384886160fc6ac'
        }

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        data = {
            "chatUuid": "",
            "chatName": "222",
            "stream": False,
            "keep": False,
            "model": self.model,
            "text": str(input)
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
            print(response.text)
            data = response.json()
            answer = data['bo']['result']
            #print(f"question: {text} answer: {answer}")
            return answer, data
        except Exception  as e:
            #print(e)
            return f"{e} chatGPT接口，出错", ""
