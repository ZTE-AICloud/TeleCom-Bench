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
OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


@MODELS.register_module()
class APIFake(BaseAPIModel):
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
                 path: str = 'llama2-api',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 api_url: str = "",
                 llama2_args: Optional[Dict] = None
                 ):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        self.llama2_args = llama2_args
        self.api_url = api_url
        print("llama2 api:"+ str(self.llama2_args))

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs
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
        # print("prompt size:%s" % len(inputs))
        # print("---------------------------")
        # print(inputs)
        # print("-------------------------------")
        length = len(inputs)
        my_list = ["A", "B", "C", "D"]
        import random
        results = random.choices(my_list, k=length)
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
        # print("***************************")
        # print(input)
        # print("******************************")
        # print("input:"+str(input))
        # if isinstance(input, str):
        #     #prompt = f"This is a conversation between user and llama, a friendly chatbot. respond in simple markdown.\n\nUser:{input} \nllama:"
        #     prompt = input
        # else:
        #     print("Prompt List:"+str(input))
        #     print("prompt size:%s" % len(input))
        #     #prompt = input['prompt']
        #     #prompt = f"This is a conversation between user and llama, a friendly chatbot. respond in simple markdown.\n\nUser:{prompt} \nllama:"

        messages = str(input)
        messages = f"User:{messages} \n llama:\n"

        data = self.llama2_args.copy()
        data['prompt'] = str(messages)
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(self.api_url, json=data, headers=headers, verify=False, stream=False)
            #print(response.content.decode('utf-8'))

            result = response.json()['content']
            print(f"*input : \n{data['prompt']} \n *Answer : {result}\n")
        except Exception as e:
            print("Exception:"+response.content.decode('utf-8'))
            return "ERROR:"+response.content.decode('utf-8')
        return result

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return 100
