from app.rag import CHUNK_OVERLAP, CHUNK_SIZE, split_text


def test_split_text_preserves_source_indexes_and_overlap():
    chunks = split_text("lesson.txt", "a" * (CHUNK_SIZE + 20))

    assert [(item.source_file, item.chunk_index) for item in chunks] == [
        ("lesson.txt", 0),
        ("lesson.txt", 1),
    ]
    assert chunks[0].text == "a" * CHUNK_SIZE
    assert chunks[1].text == "a" * (20 + CHUNK_OVERLAP)


def test_split_text_ignores_whitespace_only_documents():
    assert split_text("empty.txt", " \n\t ") == []
