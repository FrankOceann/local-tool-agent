from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from tools.note_tools import save_note
from tools.text_tools import count_words, summarize_text, upper_text
from tools.file_tools import read_file, read_files, search_files


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    function: Callable[[str], str | int]
    permission: str
    description: str
    parameters: dict[str, Any]

TEXT_PARAMETERS = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


TOOL_DEFINITIONS = [
    ToolDefinition(
        name="upper_text",
        function=upper_text,
        permission="auto",
        description="Convert text to uppercase.",
        parameters=TEXT_PARAMETERS,
    ),
    ToolDefinition(
        name="count_words",
        function=count_words,
        permission="auto",
        description="Count words in text.",
        parameters=TEXT_PARAMETERS,
    ),
    ToolDefinition(
        name="summarize_text",
        function=summarize_text,
        permission="auto",
        description="Keep the first two sentences as a short summary.",
        parameters=TEXT_PARAMETERS,
    ),
    ToolDefinition(
        name="save_note",
        function=save_note,
        permission="confirmation_required",
        description="Simulate saving a note after user confirmation.",
        parameters=TEXT_PARAMETERS,
    ),
    ToolDefinition(
        name="read_file",
        function=read_file,
        permission="auto",
        description="Read a UTF-8 text file from the allowed data directory.",
        parameters=TEXT_PARAMETERS,
    ),
    ToolDefinition(
        name="read_files",
        function=read_files,
        permission="auto",
        description=(
            "Read up to two UTF-8 text files from the allowed "
            "directory. Pass one filename per line."
        ),
        parameters=TEXT_PARAMETERS,
    ),
    ToolDefinition(
        name="search_files",
        function=search_files,
        permission="auto",
        description="Search UTF-8 text files in the allowed data directory.",
        parameters=TEXT_PARAMETERS,
    ),
]

TOOL_REGISTRY: dict[str, Callable[[str], str | int]] = {
    definition.name: definition.function
    for definition in TOOL_DEFINITIONS
}

TOOL_PERMISSIONS: dict[str, str] = {
    definition.name: definition.permission
    for definition in TOOL_DEFINITIONS
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.parameters,
        },
    }
    for definition in TOOL_DEFINITIONS
]