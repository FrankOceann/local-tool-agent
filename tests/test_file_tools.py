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