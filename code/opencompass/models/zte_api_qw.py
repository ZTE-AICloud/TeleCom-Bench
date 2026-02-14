import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union, Iterable
import json
import requests
from http import HTTPStatus

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


@MODELS.register_module()
class ZTE_qw(BaseAPIModel):

    is_api: bool = True

    def __init__(self,
                 path: str,
                 query_per_second: int = 1,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 generation_kwargs: Dict = dict()):
        super().__init__(path=path,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        ...

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 2048,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        start_time = time.time()
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        end_time = time.time()
        cost_time = end_time - start_time
        print(f"Batch 执行时间 {cost_time}  秒")
        return results

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        pass

    def _merge_messages(self, messages: List[str]) -> str:
        prompt_system = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prompt_full = prompt_system
        #prompt_full = ""

        for message in messages:
            if message["role"] == "user":
                prompt_full += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
            elif message["role"] == "assistant":
                prompt_full += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
            elif message["role"] == "system":
                prompt_full += f"<|im_start|>system\n{message['content']}<|im_end|>\n"

        prompt_full += "<|im_start|>assistant\n"
        return prompt_full

    def _post_vllm(self, api_url: str, messages: List[str], stream: bool = False) -> str:
        headers = {"User-Agent": "Test Client", "Authorization": "TEST-46542881-54d4-4096-b93d-6d5a3db326ac"}
        prompt_full = self._merge_messages(messages)
        pload_kwargs = {
            "prompt": prompt_full,
            "stream": stream,
            "n": 1,
            "best_of": 1,
            "max_tokens": 2048,
            "temperature": 0,
            "use_beam_search": False,
            "top_p": 1,
            "top_k": -1,
            "ignore_eos": False,
            "presence_penalty": 1.2,
            "frequency_penalty": 0
        }

        response = requests.post("http://10.55.33.23:31300/generate",
                                 headers=headers,
                                 json=pload_kwargs,
                                 stream=True)

        if stream:
            num_printed_lines = 0
            for h in get_streaming_response(response):
                clear_line(num_printed_lines)
                num_printed_lines = 0
                for i, line in enumerate(h):
                    num_printed_lines += 1
                    print(f"Beam candidate {i}: {line!r}", flush=True)
            output = h
        else:
            output = get_response(response)

        response_list = [i.split(prompt_full)[-1] for i in output]
        return response_list[0]

    def _generate(self, input: PromptType, max_out_len: int = 2048,):
        messages = []
        message = {}

        if isinstance(input, PromptList):
            for x in input:
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
        data = {'messages': messages}
        data.update(self.generation_kwargs)
        print("==============debugch", data)
        headers = {"Content-Type": "application/json"}

        max_num_retries = 5
        retries = 0
        err_reason = ""
        while retries < max_num_retries:
            try:
                start_time = time.time()
                answers = self._post_vllm(self.path, messages)
                end_time = time.time()
                cost_time = end_time - start_time
                print(f"接口响应时间 {cost_time}  秒")
                print(f"接口响应:{answers}")
                return answers
            except Exception as e:
                retries += 1
                err_reason += f"{retries}:{e}"
                time.sleep(5)

        result = f"Max Retries, status: Error, err_reason:{err_reason}"
        return result
