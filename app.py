"""
app.py (Streamlit UI)
시민 개발자용 간단 가이드

이 앱은 ask_rag_llm 모듈을 감싸 브라우저에서 다음을 수행합니다.
- 질문 입력 → RAG 검색 → 컨텍스트 생성 → LLM 답변 보기
- 사이드바에서 엔드포인트/모델/필터 등을 즉시 조정

실행: `streamlit run app.py`
사전 준비: `.env` 파일에 RAG/게이트웨이 키 설정
"""

import json
import traceback
from typing import Any, Dict, Optional

import streamlit as st

import ask_rag_llm as rag


st.set_page_config(page_title="RAG Q&A (Confluence)", layout="wide")  # 화면 레이아웃 설정
st.title("RAG Q&A (Confluence)")
st.caption("RAG 검색 결과를 바탕으로 LLM에 질의합니다.")


def parse_filter_json(txt: str) -> Optional[Dict[str, Any]]:
    """사이드바 텍스트 입력을 JSON dict로 파싱(오류 시 None)."""
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        st.warning("RETRIEVE_FILTER_JSON 파싱 실패. 필터를 무시합니다.")
        return None


with st.sidebar:
    # 검색/LLM 관련 설정을 사이드바에서 실시간 조정할 수 있습니다.
    st.subheader("설정")
    rag_retrieve_url = st.text_input(
        "RAG_RETRIEVE_URL", value=rag.RAG_RETRIEVE_URL, help="문서 검색 API 엔드포인트"
    )
    index_name = st.text_input(
        "RAG_INDEX_NAME", value=rag.RAG_INDEX_NAME, help="검색/업로드에 사용하는 인덱스명"
    )
    num_results = st.slider(
        "NUM_RESULT_DOC", min_value=1, max_value=20, value=int(rag.NUM_RESULT_DOC), help="검색 결과 문서 수"
    )
    max_chars_per_doc = st.slider(
        "문서당 최대 글자수", min_value=200, max_value=4000, value=1600, step=100
    )
    fields_exclude_str = st.text_input(
        "FIELDS_EXCLUDE(쉼표로 분리)", value=",".join(rag.FIELDS_EXCLUDE)
    )
    verify_ssl = st.checkbox("VERIFY_SSL", value=rag.VERIFY_SSL)
    chat_model = st.text_input("OPENAI_CHAT_MODEL", value=rag.OPENAI_CHAT_MODEL)
    st.markdown("---")
    filter_json_text = st.text_area(
        "추가 검색 필터(JSON)",
        value=rag.RETRIEVE_FILTER_JSON or "",
        height=120,
        help='예: {"file_type": ["png", "jpg"]}',
    )


question = st.text_area(
    "질문", value=rag.QUESTION or "이 문서들의 핵심 요지는 무엇인가?", height=100
)

col_run, col_reset = st.columns([1, 1])

if col_run.button("질의하기", type="primary"):
    # 모듈 런타임 설정을 UI 값으로 업데이트
    rag.RAG_RETRIEVE_URL = rag_retrieve_url
    rag.RAG_INDEX_NAME = index_name
    rag.NUM_RESULT_DOC = int(num_results)
    rag.FIELDS_EXCLUDE = [s.strip() for s in fields_exclude_str.split(",") if s.strip()]
    rag.VERIFY_SSL = bool(verify_ssl)
    rag.OPENAI_CHAT_MODEL = chat_model

    extra_filter = parse_filter_json(filter_json_text)

    try:
        with st.spinner("검색 중..."):
            result_json = rag.retrieve_rrf(question, extra_filter=extra_filter)
        st.success("검색 완료")

        with st.spinner("컨텍스트 생성 중..."):
            context_block = rag.format_context_from_results(result_json, max_chars_per_doc=max_chars_per_doc)

        if not context_block.strip():
            st.warning("검색 결과가 비어 있습니다. 질문/필터/인덱스를 확인하세요.")
        else:
            with st.spinner("LLM 질의 중..."):
                answer = rag.ask_llm(question, context_block)

            st.subheader("답변")
            st.write(answer)

            with st.expander("문서 컨텍스트 미리보기"):
                st.code(context_block)

            # 간단한 결과 메타 요약
            results = []
            if isinstance(result_json, dict):
                if "hits" in result_json and isinstance(result_json["hits"], dict) and isinstance(result_json["hits"].get("hits"), list):
                    results = [h.get("_source", {}) for h in result_json["hits"]["hits"] if isinstance(h, dict)]
                else:
                    results = result_json.get("results") or result_json.get("data") or result_json.get("docs") or []
            elif isinstance(result_json, list):
                results = result_json

            if results:
                st.subheader("검색 결과 요약")
                for i, doc in enumerate(results, 1):
                    title = (doc or {}).get("title") or (doc or {}).get("doc_title")
                    doc_id = (doc or {}).get("doc_id") or (doc or {}).get("id")
                    score = (doc or {}).get("score")
                    st.write(f"[{i}] title={title}  doc_id={doc_id}  score={score}")

            with st.expander("원본 결과(JSON)"):
                st.json(result_json)

    except Exception as e:
        st.error(f"오류: {e}")
        st.code(traceback.format_exc())

if col_reset.button("초기화"):
    st.experimental_rerun()


st.markdown("---")
st.caption("환경 변수와 인증 키는 .env를 사용합니다. secrets는 커밋하지 마세요.")
