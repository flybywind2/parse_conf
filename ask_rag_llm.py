"""
ask_rag_llm.py
시민 개발자용 간단 가이드

이 스크립트는 다음 순서로 동작합니다.
1) 질문(QUESTION)에 맞춰 RAG 검색 API에서 관련 문서를 조회합니다.
2) 검색 결과를 사람이 읽기 쉬운 컨텍스트 문자열로 변환합니다.
3) 컨텍스트만을 근거로 LLM(게이트웨이 우선, 실패 시 폴백)에 질의합니다.

환경변수(.env)에서 주로 조정하는 값
- RAG_RETRIEVE_URL, RAG_INDEX_NAME, RAG_API_KEY, PASS_KEY
- OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_TEMPERATURE
- NUM_RESULT_DOC, FIELDS_EXCLUDE, VERIFY_SSL

빠른 실행: `python ask_rag_llm.py`
질문 변경: PowerShell 예) `set QUESTION=요약해줘 && python ask_rag_llm.py`
"""

# ask_rag_llm.py
import os
import json
import time
import requests
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI  # 게이트웨이/공식 OpenAI 공용
from langchain_openai import ChatOpenAI  # 폴백용

load_dotenv()  # .env 파일을 읽어 환경변수를 주입합니다.

# =============== 환경 설정 ===============
# RAG 검색 관련 설정 (필수)
RAG_RETRIEVE_URL = os.getenv("RAG_RETRIEVE_URL", "http://apigw.server.net:8000/v2/retrieve-rrf")
RAG_INDEX_NAME   = os.getenv("RAG_INDEX_NAME", "rp-rag")
RAG_API_KEY      = os.getenv("RAG_API_KEY", "YOUR_RAG_KEY")
PASS_KEY         = os.getenv("PASS_KEY", "YOUR_PASS_KEY")

NUM_RESULT_DOC   = int(os.getenv("NUM_RESULT_DOC", "5"))
FIELDS_EXCLUDE   = os.getenv("FIELDS_EXCLUDE", "v_merge_title_content").split(",")
RETRIEVE_FILTER_JSON = os.getenv("RETRIEVE_FILTER_JSON", "")  # 예: {"example_field_name":["png"]}

VERIFY_SSL       = os.getenv("VERIFY_SSL", "true").lower() == "true"

# LLM (게이트웨이 우선) — 사내 게이트웨이 사용 시 설정, 없으면 폴백 사용
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL")   # 예: http://apigw.server.net/openai/v1
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")    # 게이트웨이/프록시 또는 OpenAI Key
X_DEP_TICKET      = os.getenv("X_DEP_TICKET", "")
SYSTEM_NAME       = os.getenv("SYSTEM_NAME", "System-Name.")
USER_ID           = os.getenv("USER_ID", "ID")
USER_TYPE         = os.getenv("USER_TYPE", "Type")

OPENAI_CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

QUESTION           = os.getenv("QUESTION", "이 문서들의 핵심 요지는 무엇인가?")

TIMEOUT_SEC = 60          # HTTP 요청 타임아웃(초)
MAX_RETRIES = 3           # 재시도 횟수
RETRY_BACKOFF_SEC = 2     # 재시도 백오프 기본(초)
# ========================================

def build_headers() -> Dict[str, str]:
    """RAG 검색용 공통 헤더 구성."""
    return {"Content-Type":"application/json","x-dep-ticket":PASS_KEY,"api-key":RAG_API_KEY}

def post_with_retries(url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> requests.Response:
    """간단한 재시도 로직이 포함된 POST 호출 도우미."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT_SEC, verify=VERIFY_SSL)
            if 200 <= resp.status_code < 300: return resp
            if resp.status_code in (429,500,502,503,504):
                print(f"[경고] {url} {resp.status_code} (재시도 {attempt}/{MAX_RETRIES}) -> {resp.text[:300]}")
            else:
                return resp
        except requests.RequestException as e:
            last_exc = e
            print(f"[예외] {e} (재시도 {attempt}/{MAX_RETRIES})")
        time.sleep(RETRY_BACKOFF_SEC * attempt)
    if last_exc: raise last_exc
    raise RuntimeError("요청 실패(원인 불명)")

def retrieve_rrf(query_text: str, extra_filter: Optional[Dict[str,Any]] = None) -> Any:
    """RAG 엔드포인트에서 관련 문서를 조회합니다.
    - query_text: 질문/키워드 문자열
    - extra_filter: 추가 필터(JSON dict). UI나 .env에서 전달 가능
    반환값 스키마는 서버 구현에 따라 달라질 수 있습니다.
    """
    headers = build_headers()
    payload: Dict[str,Any] = {
        "index_name": RAG_INDEX_NAME,
        "query_text": query_text,
        "num_result_doc": NUM_RESULT_DOC,
        "fields_exclude": FIELDS_EXCLUDE,
    }
    if RETRIEVE_FILTER_JSON.strip():
        try:
            payload["filter"] = json.loads(RETRIEVE_FILTER_JSON)
        except Exception:
            print("[경고] RETRIEVE_FILTER_JSON 파싱 실패 → 무시")
    if extra_filter:
        payload.setdefault("filter", {}).update(extra_filter)

    print(f"\n[검색] {RAG_RETRIEVE_URL}  query='{query_text}'")
    resp = post_with_retries(RAG_RETRIEVE_URL, headers, payload)
    print(f"[응답] {resp.status_code}")

    try:
        obj = resp.json()
    except Exception:
        print(resp.text[:500]); raise

    # 리스트로도, dict로도 올 수 있음 — 일단 그대로 반환하고 포맷터에서 처리
    return obj

def format_context_from_results(result_json: Any, max_chars_per_doc: int = 1600) -> str:
    """
    다양한 스키마 처리:
    - ES: {"hits":{"hits":[{"_source": {...}}]}}
    - 일반: {"results":[...]}, {"data":[...]}, {"docs":[...]} ...
    - 리스트 루트
    결과를 제목/ID/점수/본문 일부로 묶어 LLM 컨텍스트 블록을 만들어 줍니다.
    """
    results = []

    # (A) ES 스타일
    if isinstance(result_json, dict) and "hits" in result_json:
        hits = result_json.get("hits", {})
        if isinstance(hits, dict) and "hits" in hits and isinstance(hits["hits"], list):
            for h in hits["hits"]:
                if isinstance(h, dict):
                    src = h.get("_source", {})
                    if src:
                        results.append(src)

    # (B) 일반 컨테이너/리스트
    if not results:
        if isinstance(result_json, list):
            results = result_json
        elif isinstance(result_json, dict):
            for key in ("results", "data", "docs", "items"):
                v = result_json.get(key)
                if isinstance(v, list):
                    results = v
                    break
            if not results and isinstance(result_json.get("results"), dict):
                for key in ("items","hits","documents"):
                    v = result_json["results"].get(key)
                    if isinstance(v, list):
                        results = v
                        break

    # 문자열 조립
    lines = []
    for i, doc in enumerate(results, 1):
        def get(d: dict, *keys):
            return next((d[k] for k in keys if isinstance(d, dict) and k in d and d[k] is not None), "")
        title = str(get(doc, "title", "doc_title", "name") or f"Doc{i}")
        doc_id = str(get(doc, "doc_id", "id", "uuid") or "")
        score  = doc.get("score") if isinstance(doc, dict) else None
        content = get(doc, "content", "text", "snippet", "preview", "body", "description", "v_merge_title_content")
        if not content:
            try:
                str_fields = [(k, v) for k, v in doc.items() if isinstance(v, str)]
                if str_fields:
                    content = max(str_fields, key=lambda kv: len(kv[1]))[1]
            except Exception:
                content = ""
        content = str(content)[:max_chars_per_doc].strip()
        head = f"[{i}] {title}"
        parts = []
        if doc_id: parts.append(f"doc_id={doc_id}")
        if score is not None: parts.append(f"score={score}")
        if parts: head += " (" + ", ".join(parts) + ")"
        lines.append(head + ("\n" + content if content else ""))
    return "\n\n".join(lines)

def _gateway_client() -> Optional[OpenAI]:
    if not OPENAI_BASE_URL:
        return None
    headers = {
        "x-dep-ticket": X_DEP_TICKET,
        "Send-System-Name": SYSTEM_NAME,
        "User-Id": USER_ID,
        "User-Type": USER_TYPE,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, default_headers=headers)

def ask_llm(question: str, context_block: str) -> str:
    """게이트웨이(OpenAI 호환) 우선 → 실패 시 ChatOpenAI 폴백.
    컨텍스트 외 지식은 사용하지 않도록 시스템 메시지를 고정합니다.
    """
    client = _gateway_client()
    sys_msg = (
        "당신은 아래 '문서 컨텍스트'만을 신뢰하여 한국어로 정확하고 간결하게 답합니다. "
        "가능하면 출처를 대괄호 번호로 표시하세요(예: [1], [2]). "
        "컨텍스트에 없는 내용은 '제공된 문서에서 찾을 수 없음'이라고 답하세요."
    )
    user_msg = f"# 질문\n{question}\n\n# 문서 컨텍스트\n{context_block}"

    if client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_CHAT_MODEL,
                    temperature=OPENAI_TEMPERATURE,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    print(f"[경고] 게이트웨이 LLM 실패 최종: {e}")
                time.sleep(RETRY_BACKOFF_SEC * attempt)

    # 폴백: 공식 OpenAI 경유
    try:
        llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=OPENAI_TEMPERATURE)
        prompt = f"{sys_msg}\n\n{user_msg}\n\n답변:"
        return llm.invoke(prompt).content
    except Exception as e:
        return f"[LLM 호출 오류] {e}"

def main():
    """스크립트 진입점: 검색→컨텍스트 구성→LLM 질의→결과/메타 출력."""
    question = os.getenv("QUESTION", "이 문서들의 핵심 요지는 무엇인가?")

    # 1) 검색
    result_json = retrieve_rrf(question, extra_filter=None)

    # 2) 컨텍스트 생성
    context_block = format_context_from_results(result_json, max_chars_per_doc=1600)

    # 3) 컨텍스트 검증 & 안내
    if not context_block.strip():
        print("[안내] 검색 결과가 비었습니다. 질문/필터/인덱스를 확인하세요.")
        if isinstance(result_json, dict):
            total = (result_json.get("hits", {}).get("total", {}) or {}).get("value")
            print(f"hits.total.value = {total}")
            print("top-level keys:", list(result_json.keys())[:20])
        else:
            print("raw type:", type(result_json))
        return

    # 4) LLM 답변
    answer = ask_llm(question, context_block)
    print("\n=== LLM 답변 ===\n", answer)

    # 5) 출처 메타(요약)
    results = []
    if isinstance(result_json, dict):
        if "hits" in result_json and isinstance(result_json["hits"], dict) and isinstance(result_json["hits"].get("hits"), list):
            results = [h.get("_source", {}) for h in result_json["hits"]["hits"] if isinstance(h, dict)]
        else:
            results = result_json.get("results") or result_json.get("data") or result_json.get("docs") or []
    elif isinstance(result_json, list):
        results = result_json

    if results:
        print("\n=== 검색 원문 메타(요약) ===")
        for i, doc in enumerate(results, 1):
            title = (doc or {}).get("title") or (doc or {}).get("doc_title")
            doc_id = (doc or {}).get("doc_id") or (doc or {}).get("id")
            score  = (doc or {}).get("score")
            print(f"[{i}] title={title}  doc_id={doc_id}  score={score}")

if __name__ == "__main__":
    main()
