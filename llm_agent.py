import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tools import count_words, summarize_text, upper_text


MISSING_KEY_MESSAGE = "未检测到 DEEPSEEK_API_KEY，请在 .env 中配置后重试。"
API_CALL_ERROR_MESSAGE = "调用模型服务失败，请稍后重试。"
MULTIPLE_TOOL_CALLS_MESSAGE = "模型一次请求了多个工具，当前仅支持一个工具调用。"
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_TOOL_CALLS = 3
SYSTEM_PROMPT = (
    "你是一个由 DeepSeek 驱动的本地 Tool Agent。"
    "你不是 Claude、Anthropic 或 OpenAI 的官方助手。"
    "你目前可以使用 upper_text（把文本转为大写）、"
    "count_words（统计英文单词）和 summarize_text（保留文本前两句作为摘要）"
    "三个本地工具。"
    "没有对应工具时，不要声称你可以联网、查询实时数据、读取文件或执行外部操作。"
)
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
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Keep the first two sentences as a short summary.",
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
        self.tool_registry = {
            "upper_text": upper_text,
            "count_words": count_words,
            "summarize_text": summarize_text,
}
        if self.client is None and self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)

    def run(self, request: str) -> str:
        if not self.api_key:
            return MISSING_KEY_MESSAGE

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]

        for _ in range(MAX_TOOL_CALLS):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception:
                return API_CALL_ERROR_MESSAGE

            assistant_message = response.choices[0].message
            if not assistant_message.tool_calls:
                return assistant_message.content or "模型没有返回可显示的回答。"

            if len(assistant_message.tool_calls) != 1:
                return MULTIPLE_TOOL_CALLS_MESSAGE

            tool_call = assistant_message.tool_calls[0]
            result, error = self._run_tool(
                tool_call.function.name,
                tool_call.function.arguments,
            )
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

        try:
            final_response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
        except Exception:
            return API_CALL_ERROR_MESSAGE

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
