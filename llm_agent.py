import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import count_words, upper_text


MISSING_KEY_MESSAGE = "未检测到 DEEPSEEK_API_KEY，请在 .env 中配置后重试。"
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "upper_text",
            "description": "Convert text to uppercase.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count words in text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]


class LLMToolAgent:
    def __init__(self, client: object | None = None, api_key: str | None = None):
        load_dotenv()
        self.client = client
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.tool_registry = {"upper_text": upper_text, "count_words": count_words}
        if self.client is None and self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)

    def run(self, request: str) -> str:
        if not self.api_key:
            return MISSING_KEY_MESSAGE

        messages = [{"role": "user", "content": request}]
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            return assistant_message.content or "模型没有返回可显示的回答。"

        tool_call = assistant_message.tool_calls[0]
        result, error = self._run_tool(tool_call.function.name, tool_call.function.arguments)
        if error:
            return error

        messages.append(assistant_message)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        )
        final_response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        return final_response.choices[0].message.content or "模型没有返回可显示的回答。"

    def _run_tool(
        self, name: str, arguments_json: str
    ) -> tuple[str | int | None, str | None]:
        if name not in self.tool_registry:
            return None, f"模型请求了不支持的工具: {name}"

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            return None, "模型返回的工具参数不是有效 JSON。"

        if not isinstance(arguments, dict) or not isinstance(arguments.get("text"), str):
            return None, "工具参数 text 必须是字符串。"

        return self.tool_registry[name](arguments["text"]), None
