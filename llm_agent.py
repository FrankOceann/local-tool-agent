import os

from dotenv import load_dotenv


MISSING_KEY_MESSAGE = "未检测到 DEEPSEEK_API_KEY，请在 .env 中配置后重试。"


class LLMToolAgent:
    def __init__(self, client: object | None = None, api_key: str | None = None):
        load_dotenv()
        self.client = client
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")

    def run(self, request: str) -> str:
        if not self.api_key:
            return MISSING_KEY_MESSAGE
        return ""
