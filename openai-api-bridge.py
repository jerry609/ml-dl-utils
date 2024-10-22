import aiohttp
from autogen_core.components.models._openai_client import BaseOpenAIChatCompletionClient
import requests

class ThirdPartyChatCompletionClient(BaseOpenAIChatCompletionClient):
    def __init__(self, **kwargs):
        if "model" not in kwargs:
            raise ValueError("model is required for ThirdPartyChatCompletionClient")

        model_capabilities = kwargs.pop("model_capabilities", None)
        self.api_key = kwargs.get("api_key")
        self.model = kwargs.get("model")
        self.url = "https://api.gptgod.online/v1/chat/completions"

        # 创建 create_args，至少包含 'model' 键
        create_args = {"model": self.model}

        # 保存原始配置
        self._raw_config = kwargs.copy()
        super().__init__(None, create_args, model_capabilities)

    async def agenerate(self, messages, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    # 根据需要提取模型的回复
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"请求失败，状态码: {response.status}")
                    return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
