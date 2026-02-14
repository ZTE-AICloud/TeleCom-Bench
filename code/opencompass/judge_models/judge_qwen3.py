import re
import time

import requests


class Qwen3:
    def __init__(self, base_url="http://10.55.42.83:31059/v1/chat/completions", model="Qwen3-235B-A22B-Instruct-2507",
                 retries=1):
        self.retries = retries
        self.base_url = base_url
        self.model = model
        self.timeout = 15 * 60
        self.headers = {
            "Content-Type": "application/json"
        }
        self.temperature = 0

    def chat(self, message, stream=False, detail=False):
        if isinstance(message, list):
            messages = message
        else:
            messages = [
                {
                    "content": message,
                    "role": "user"
                }
            ]

        payload = {
            "model": self.model,
            "stream": stream,
            "max_tokens": 8192,
            "temperature": self.temperature,
            "messages": messages
        }
        for attempt in range(self.retries):
            try:
                if stream:
                    with requests.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout,
                                       stream=True) as response:
                        # response.raise_for_status()
                        content = r""
                        for line in response.iter_lines():
                            print(content)
                            result = str(line, encoding='utf-8')
                            print(result)
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
                                        result = re.search(r'"content":"(.*?)"', result).group(1)
                                        # result = result.replace("\\", "\n")
                                        print(result)
                                        content += result
                                    else:
                                        content += ""
                                except Exception as e:
                                    print(e)
                                    continue
                        return eval('f"' + content + '"')
                else:
                    response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout)
                    print(response.text)
                    response_data = response.json()
                    print(f"Response data: {response_data}")
                    print(response_data['choices'][-1])
                    print(f"prompt:{message}")
                    content = response_data['choices'][-1]['message']['content'].encode('utf-8').decode('utf-8')
                    print(f"content:{content}")
                    return content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return "LLM ERROR"

    def predict(self, input_text, **kwargs):
        # print(f"input_text:{input_text}  kwargs:{kwargs}")
        return self.chat(message=input_text)

    def __call__(self, input_text, **kwargs):
        ans = self.predict(input_text, **kwargs)
        print(f"问题{input_text}\n模型回答:{ans}")
        return ans

# 示例使用
# if __name__ == "__main__":
#     #qwen_110b_int4_key = "fastgpt-sIe6HsiS0Bg5LASc09OkvXesxGgQuTRrSSrZe4aAgD2V2SPHMUdRzs44h1vRBeJi"
#     #qwen2_72b_int4_key = 'fastgpt-sw7h2HDKpT6gvlFaKfXjxxGiFTXF9n0HdmzCkTHcFJDvsiTMpPqH'
#     client = VllmOpenAi()
#     response = client.chat(message="出一个4选1的选择题，并给出答案和解释", stream=False)
#     print(response)
