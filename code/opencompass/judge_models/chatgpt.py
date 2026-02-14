import requests
import json
import random

class ChatGPT:
    url = "https://rdcloud.zte.com.cn/zte-studio-ai-platform/openapi/v1/chat"
    headers = {
        'Authorization': 'Bearer 80366441f8a9480eb312943a0a411054-23baa365c70b49a5afc7520bc57ae299',
        'Content-Type': 'application/json',
        'X-Emp-No': '00054055',
        'X-Auth-Value': '5a856d5b6fc3b0a431384886160fc6ac'
    }
    def __init__(self, model="gpt-4", name = "test") -> None:
        self.name = name
        self.model = model
        self.data = {
            "chatUuid": "",
            "chatName": "222",
            "stream": False,
            "keep": False,
            "model": model
        }
    
    def chat(self, text):
        try:
            self.data['text'] = text
            response = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
            print(response.text)
            data = response.json()
            answer = data['bo']['result']
            #print(f"question: {text} answer: {answer}")
            return answer
        except Exception  as e:
            #print(e)
            return f"{e} chatGPT接口，出错"
    
    def clear_history(self):
        pass

    def predict(self, input_text, **kwargs):
        print(f"input_text:{input_text}  kwargs:{kwargs}")
        return self.chat(input_text)

    
    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)
    

if __name__ == "__main__":
    client = ChatGPT(model="gpt-4-turbo")
    print(client.chat("hi"))
    

    client = ChatGPT(model="gpt-4o")
    print(client.chat("hi"))

    client = ChatGPT(model="gpt-4o-mini")
    print(client.chat("hi"))

    client = ChatGPT(model="gpt-4")
    print(client.chat("hi"))
