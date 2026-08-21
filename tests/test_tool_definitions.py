from app.tool_schemas import TOOL_SCHEMAS
from tools import registry


def test_each_tool_definition_is_exposed_to_the_agent_consistently():
    """A new tool definition must produce its callable, permission, and schema."""
    definitions = registry.TOOL_DEFINITIONS

    assert [definition.name for definition in definitions] == [
        "upper_text",
        "count_words",
        "summarize_text",
        "save_note",
    ]

    for definition in definitions:
        assert registry.TOOL_REGISTRY[definition.name] is definition.function
        assert registry.TOOL_PERMISSIONS[definition.name] == definition.permission

    assert [schema["function"]["name"] for schema in TOOL_SCHEMAS] == [
        definition.name for definition in definitions
    ]
