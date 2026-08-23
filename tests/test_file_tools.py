from pathlib import Path
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

def test_search_files_returns_matching_filenames(tmp_path, monkeypatch):
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

    assert result == "agent_basics.txt"

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

def test_search_files_ignores_english_letter_case(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)
    (tmp_path / "agent_basics.txt").write_text(
        "Agent 可以调用工具。",
        encoding="utf-8",
    )

    result = file_tools.search_files("agent")

    assert result == "agent_basics.txt"

def test_search_files_matches_filenames_case_insensitively(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)
    (tmp_path / "agent_safety.txt").write_text(
        "文件访问必须校验权限。",
        encoding="utf-8",
    )

    result = file_tools.search_files("SAFETY")

    assert result == "agent_safety.txt"

def test_search_files_rejects_a_blank_keyword(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)

    result = file_tools.search_files("   ")

    assert result == "搜索关键词不能为空。"

def test_search_files_sorts_results_when_directory_returns_them_out_of_order(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)
    (tmp_path / "alpha.txt").write_text(
        "包含关键词。",
        encoding="utf-8",
    )
    (tmp_path / "zeta.txt").write_text(
        "也包含关键词。",
        encoding="utf-8",
    )

    original_glob = Path.glob
    monkeypatch.setattr(
        Path,
        "glob",
        lambda directory, pattern: reversed(
            list(original_glob(directory, pattern))
        ),
    )

    result = file_tools.search_files("关键词")

    assert result == "alpha.txt\nzeta.txt"

def test_search_files_returns_only_the_first_three_sorted_results(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)

    for filename in [
        "delta.txt",
        "alpha.txt",
        "charlie.txt",
        "bravo.txt",
    ]:
        (tmp_path / filename).write_text(
            "包含关键词。",
            encoding="utf-8",
        )

    result = file_tools.search_files("关键词")

    assert result == "alpha.txt\nbravo.txt\ncharlie.txt"

def test_search_files_prioritizes_filename_matches_over_content_matches(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)

    (tmp_path / "alpha_notes.txt").write_text(
        "agent agent agent",
        encoding="utf-8",
    )
    (tmp_path / "z_agent_guide.txt").write_text(
        "普通资料。",
        encoding="utf-8",
    )

    result = file_tools.search_files("agent")

    assert result == "z_agent_guide.txt\nalpha_notes.txt"

def test_search_files_ranks_content_matches_by_occurrence_count(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(file_tools, "DATA_DIRECTORY", tmp_path)

    (tmp_path / "alpha_notes.txt").write_text(
        "agent",
        encoding="utf-8",
    )
    (tmp_path / "zeta_notes.txt").write_text(
        "agent agent agent",
        encoding="utf-8",
    )

    result = file_tools.search_files("agent")

    assert result == "zeta_notes.txt\nalpha_notes.txt"