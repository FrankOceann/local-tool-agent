from pathlib import Path


DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
MAX_SEARCH_RESULTS = 3

def read_file(filename: str) -> str:
    allowed_directory = DATA_DIRECTORY.resolve()
    file_path = (allowed_directory / filename).resolve()

    if not file_path.is_relative_to(allowed_directory):
        return "不允许读取 data 目录外的文件。"

    if not file_path.exists():
        return "文件不存在。"

    return file_path.read_text(encoding="utf-8")

def search_files(query: str) -> str:
    normalized_query = query.strip().casefold()

    if not normalized_query:
        return "搜索关键词不能为空。"

    results = []

    for file_path in DATA_DIRECTORY.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        normalized_content = content.casefold()

        matches_filename = normalized_query in file_path.name.casefold()
        content_match_count = normalized_content.count(normalized_query)
        matches_content = content_match_count > 0

        if matches_filename or matches_content:
            priority = 0 if matches_filename else 1
            results.append(
                (priority, -content_match_count, file_path.name)
            )

    if not results:
        return "没有找到匹配内容。"

    return "\n".join(
        filename
        for _, _, filename in sorted(results)[:MAX_SEARCH_RESULTS]
    )
