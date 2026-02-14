from .base_api import BaseAPIModel
from typing import List, Union, Dict, Optional

from concurrent.futures import ThreadPoolExecutor
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
import requests
import json

PromptType = Union[PromptList, str]

@MODELS.register_module()
class Llama2API(BaseAPIModel):
    is_api: bool = True

    def __init__(self,
                path: str = 'llama2_70b_api',
                max_seq_len: int = 4096,
                query_per_second: int = 1,
                retry: int = 2,
                key: Union[str, List[str]] = 'ENV',
                api_url: str = '',
                mode: str = 'none',
                org: Optional[Union[str, List[str]]] = None,
                meta_template: Optional[Dict] = None,
                temperature: Optional[float] = None):
        
        super().__init__(path=path,
                            max_seq_len=max_seq_len,
                            meta_template=meta_template,
                            query_per_second=query_per_second,
                            retry=retry)
        self.temperature = temperature
        self.api_url = api_url

    def generate(
        self,
        inputs,
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs."""

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results
    
    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        if not isinstance(input, str):
            return "ERROR: wrong input"
        messages = {
            "llm_service": "llama_cpp_general_70B",
            "user": "10293467",
            "ip": "10.118.19.109",
            "request": {
                "stream": False,
                "max_tokens": 400,
                "temperature": 0.7,
                "stop": [
                    "</s>",
                    "llama:",
                    "User:"
                ],
                "top_k": 40,
                "top_p": 0.5,
            }
        }
        messages["request"]['prompt'] = f"User: {input} \nllama:"
        headers = {
            'Content-Type': 'application/json',
        }            
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()
            try:
                # response = requests.post(self.api_url, json=messages, headers=headers, verify=False, stream=False)
                raw_response = requests.post(self.api_url, json=messages, headers=headers, verify=False)
            except requests.ConnectionError:
                self.logger.warn('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()        
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                    str(raw_response.content))
                continue            
            try:
                self.logger.info("\n===============\n")
                self.logger.info(f"Prompt: {json.dumps(messages, ensure_ascii=False)}\n")
                self.logger.info(f"Response: {response}\n")
                self.logger.info("\n===============\n")
                return response['text'].strip()
            except Exception as e:
                err_msg = f"Exception: {e} resp: {response.content.decode('utf-8')}\n"
                self.logger.error(err_msg)
                return err_msg
            return result
        raise RuntimeError('Calling LLAMA2_70B_API failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')        


    # def get_token_len(self, prompt: str) -> int:
    #     """Get lengths of the tokenized string."""
    #     pass        