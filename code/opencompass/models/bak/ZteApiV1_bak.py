import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from datetime import datetime

from openai import OpenAI

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]

def print_exc_with_timestamp():
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} Exception traceback:")
    traceback.print_exc()

@MODELS.register_module()
class ZteApiV1(BaseAPIModel):
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
                 max_seq_len: int = 1024,
                 query_per_second: int = 1,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 stream: bool = True
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
        self.model = api_data['model']
        self.api_key = self.headers.get("Authorization", "Fake_Authorization")

        self.base_url = api_url.split("v1")[0] + "v1"
        self.temperature = 0.7

    def generate(
            self,
            inputs: List[str or PromptList],
            max_out_len: int = 512,
            temperature: float = 0,
            **kwargs
    ) -> List[str]:
        start_time = time.time()

        batch_size = len(inputs)
        # print(f"inputs 类型 {type(inputs)} {inputs}")
        print(f"Batch_size:{batch_size}")
        with ThreadPoolExecutor(max_workers=256) as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)
                             ))
        end_time = time.time()
        cost_time = end_time - start_time
        print(f"Batch 执行时间 {cost_time}  秒")
        return results

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float
                  ) -> str:
        # print(f"input: {input}  {type(input)}")
        messages = []
        message = {}
        # for x in input:
        #     print(f"x : {x} xtype {type(x)}")

        if isinstance(input, PromptList):
            for x in input:
                # print(f"x : {x}")
                if x['role'] == "HUMAN":
                    message = {"role": "user", "content": x['prompt']}
                elif x['role'] == "BOT":
                    message = {"role": "assistant", "content": x['prompt']}
                elif x['role'] == "SYSTEM":
                    message = {"role": "system", "content": x['prompt']}
                messages.append(message)
        else:
            message = {'role': 'user', 'content': input}
            messages.append(message)

        data = self.api_data.copy()
        # data['prompt'] = str(messages)
        # data['messages'][0]['content'] = str(messages)
        data['messages'] = messages
        MAX_RETRIES = 3  # 可根据需要设置最大重试次数
        retries = 0
        err_reason = ""
        while retries < MAX_RETRIES:
            try:
                if self.stream:
                    result = self.generate_stream(messages)
                else:
                    result = self.generate_no_stream(messages)
                # print(self.temperature)
                print(f"模型回答:{result}")
                return result
            except Exception as e:
                print_exc_with_timestamp()
                retries += 1
                err_reason += f"{retries}:{e}"
                time.sleep(5)
        result = f"Max Retries, status: Error, err_reason:{err_reason}"
        return result

    def generate_no_stream(self, messages):
        client = OpenAI(
            api_key=f"Bearer {self.api_key}",
            base_url=self.base_url,
            timeout=5 * 60,
            default_headers=self.headers
        )
        completion = client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature = self.temperature,
            stream=False
        )
        ans = completion.choices[0].message.content
        return ans

    def generate_stream(self, messages):
        client = OpenAI(
            api_key=f"Bearer {self.api_key}",
            base_url=self.base_url,
            timeout=5 * 60,
            default_headers=self.headers
        )
        print(self.temperature)
        print("请求消息")
        stream = client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature = self.temperature,
            stream=True
        )
        text = ""
        for chunk in stream:
            # print(chunk.choices[0].delta.content or "", end="")
            text += chunk.choices[0].delta.content or ""
        return text
