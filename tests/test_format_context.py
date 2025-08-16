"""간단한 포맷터 단위 테스트

네트워크 호출 없이 format_context_from_results의 핵심 동작만 검증합니다.
"""

import json

from ask_rag_llm import format_context_from_results


def test_format_context_es_like():
    es_payload = {
        "hits": {
            "hits": [
                {"_source": {"title": "Doc A", "doc_id": "a1", "score": 1.0, "content": "Alpha"}},
                {"_source": {"title": "Doc B", "doc_id": "b2", "score": 0.9, "content": "Beta"}},
            ]
        }
    }
    out = format_context_from_results(es_payload, max_chars_per_doc=50)
    assert "[1] Doc A (doc_id=a1, score=1.0)" in out
    assert "Alpha" in out
    assert "[2] Doc B (doc_id=b2, score=0.9)" in out


def test_format_context_generic_list():
    payload = [
        {"name": "Item 1", "id": "x1", "text": "Lorem ipsum dolor"},
        {"title": "Item 2", "uuid": "x2", "body": "Dolor sit amet"},
    ]
    out = format_context_from_results(payload, max_chars_per_doc=50)
    assert "[1] Item 1 (doc_id=x1)" in out
    assert "Lorem ipsum" in out
    assert "[2] Item 2 (doc_id=x2)" in out
