from tools import file_tools



def test_read_file_returns_text_from_the_allowed_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)
    (tmp_path / "lesson.txt").write_text("Agent 学习笔记", encoding="utf-8")

    result = file_tools.read_file("lesson.txt")

    assert result == "Agent 学习笔记"

def test_read_file_rejects_a_path_outside_the_allowed_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)
    (tmp_path.parent / "secret.txt").write_text("不应被读取", encoding="utf-8")

    result = file_tools.read_file("../secret.txt")

    assert result == "不允许读取 data 目录外的文件。"

def test_read_file_returns_a_clear_message_when_the_file_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)

    result = file_tools.read_file("missing.txt")

    assert result == "文件不存在。"

def test_search_files_returns_the_text_of_matching_files(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)

    (tmp_path / "agent_basics.txt").write_text(
        "Agent 可以调用工具。",
        encoding="utf-8",
    )
    (tmp_path / "python_notes.txt").write_text(
        "Python 负责文件处理。",
        encoding="utf-8",
    )

    result = file_tools.search_files("工具")

    assert result == "agent_basics.txt: Agent 可以调用工具。"

def test_search_files_returns_a_clear_message_when_nothing_matches(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)
    (tmp_path / "agent_basics.txt").write_text(
        "Agent 可以调用工具。",
        encoding="utf-8",
    )

    result = file_tools.search_files("数据库")

    assert result == "没有找到匹配内容。"