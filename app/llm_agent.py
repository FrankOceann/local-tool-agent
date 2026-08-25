import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.config import (
    API_CALL_ERROR_MESSAGE,
    BASE_URL,
    MAX_TOOL_CALLS,
    MISSING_KEY_MESSAGE,
    MODEL_NAME,
    SYSTEM_PROMPT,
    TOOL_CALL_LIMIT_MESSAGE,
    UNSTRUCTURED_TOOL_CALL_MESSAGE,
)
from app.tool_schemas import TOOL_SCHEMAS
from tools.registry import TOOL_PERMISSIONS, TOOL_REGISTRY


class LLMToolAgent:
    def __init__(self, client: object | None = None, api_key: str | None = None):
        load_dotenv()
        self.client = client
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.tool_registry = TOOL_REGISTRY.copy()
        self.pending_tool_call: dict[str, str] | None = None

        if self.client is None and self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)

    def run(self, request: str) -> str:
        pending_result = self._handle_pending_tool_call(request)
        if pending_result is not None:
            return pending_result

        if not self.api_key:
            return MISSING_KEY_MESSAGE

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]

        executed_tool_calls = 0

        while executed_tool_calls < MAX_TOOL_CALLS:
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
                if assistant_message.content and (
                    "<tool_calls>" in assistant_message.content
                    or (
                        "DSML" in assistant_message.content
                        and "tool_calls" in assistant_message.content
                    )
                ):
                    return UNSTRUCTURED_TOOL_CALL_MESSAGE
                return assistant_message.content or "模型没有返回可显示的回答。"

            if len(assistant_message.tool_calls) > MAX_TOOL_CALLS - executed_tool_calls:
                return TOOL_CALL_LIMIT_MESSAGE

            requires_confirmation = [
                tool_call
                for tool_call in assistant_message.tool_calls
                if TOOL_PERMISSIONS.get(tool_call.function.name)
                == "confirmation_required"
            ]
            if requires_confirmation and len(assistant_message.tool_calls) > 1:
                return (
                    "同一轮请求包含需要确认的工具，"
                    "当前不支持与其他工具混合执行。"
                )

            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                name = tool_call.function.name
                arguments_json = tool_call.function.arguments

                arguments, error = self._parse_tool_arguments(
                    name,
                    arguments_json,
                )
                if error:
                    return error

                if TOOL_PERMISSIONS[name] == "confirmation_required":
                    self.pending_tool_call = {
                        "name": name,
                        "arguments_json": arguments_json,
                        "tool_call_id": tool_call.id,
                    }
                    return (
                        f"操作需要确认：将模拟保存笔记“{arguments['text']}”。"
                        "请输入“确认”或“取消”。"
                    )

                result, error = self._run_tool(name, arguments_json)
                if error:
                    return error

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
                executed_tool_calls += 1

        try:
            final_response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
        except Exception:
            return API_CALL_ERROR_MESSAGE

        return final_response.choices[0].message.content or "模型没有返回可显示的回答。"

    def _handle_pending_tool_call(self, request: str) -> str | None:
        cleaned_request = request.strip()

        if self.pending_tool_call is None:
            if cleaned_request in {"确认", "取消"}:
                return "当前没有待确认的操作。"
            return None

        if cleaned_request == "取消":
            self.pending_tool_call = None
            return "已取消待确认的操作。"

        if cleaned_request != "确认":
            return "当前有待确认的操作，请输入“确认”或“取消”。"

        pending = self.pending_tool_call
        result, error = self._run_tool(
            pending["name"],
            pending["arguments_json"],
        )
        if error:
            return error

        self.pending_tool_call = None
        return str(result)

    def _parse_tool_arguments(
        self, name: str, arguments_json: str
    ) -> tuple[dict[str, str] | None, str | None]:
        if name not in self.tool_registry:
            return None, f"模型请求了不支持的工具: {name}"

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            return None, "模型返回的工具参数不是有效 JSON。"

        if not isinstance(arguments, dict) or not isinstance(arguments.get("text"), str):
            return None, "工具参数 text 必须是字符串。"

        return arguments, None

    def _run_tool(
        self, name: str, arguments_json: str
    ) -> tuple[str | int | None, str | None]:
        arguments, error = self._parse_tool_arguments(name, arguments_json)
        if error:
            return None, error

        return self.tool_registry[name](arguments["text"]), None
