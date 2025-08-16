# 저장소 가이드라인 (Repository Guidelines)

## 프로젝트 구조와 모듈
- `ask_rag_llm.py`: RAG 검색 API에서 문서를 조회하고 컨텍스트를 만들어 LLM에 질의합니다. 답변과 간단한 출처 메타를 출력합니다.
- `ingest_to_rag.py`: Confluence 페이지를(선택적 재귀) 수집해 HTML→Markdown 정규화, 링크 절대화, 선택적 이미지 설명을 추가하고 RAG 인덱스로 업로드합니다.
- `.env`(로컬 생성): 두 스크립트에서 사용하는 환경변수와 키를 보관합니다.

## 빌드·실행·개발 명령
- 가상환경(Windows): `python -m venv .venv && .venv\\Scripts\\activate`
- 의존성 설치: `pip install -r requirements.txt`
- 수집 실행(쉼표로 pageId 나열): `python ingest_to_rag.py 12345,67890`
- 질의 실행: `python ask_rag_llm.py`

## 설정 팁(.env 예시)
```
RAG_RETRIEVE_URL=http://apigw.server.net:8000/v2/retrieve-rrf
RAG_INSERT_URL=http://apigw.server.net:8000/v2/insert-doc
RAG_INDEX_NAME=rp-rag
RAG_API_KEY=...
PASS_KEY=...
VERIFY_SSL=true
OPENAI_BASE_URL=http://apigw.server.net/openai/v1
OPENAI_API_KEY=...
X_DEP_TICKET=...
SYSTEM_NAME=System-Name.
USER_ID=ID
USER_TYPE=Type
CONF_BASE_URL=https://confluence.example.com
CONF_USER=you@example.com
CONF_TOKEN=confluence_api_token
CONF_PAGE_IDS=12345,67890
CONF_RECURSIVE=true
CONF_MAX_DEPTH=3
VISION_ENABLE=false
```

## 코딩 스타일·네이밍
- Python PEP 8, 들여쓰기 4칸. 함수/모듈은 `snake_case`, 클래스는 `CamelCase`.
- 환경변수는 파일 상단에서 명시적으로 로드, 작은 함수 단위 유지.
- 사용자 메시지/로그는 기존 한국어 톤을 유지합니다.

## 테스트 가이드
- 기본 테스트는 없음. 작은 `CONF_PAGE_IDS`로 업로드 로그를 확인한 뒤 `ask_rag_llm.py`로 질의하여 검증하세요.
- 단위 테스트 추가 시 `pytest` 권장, 파일은 `tests/test_*.py` 패턴을 사용하세요.

## 커밋·PR 가이드
- 커밋: Conventional Commits 권장(예: `feat: add image OCR`, `fix: retry on 502`). 명령형, 작은 단위로.
- PR: 목적, 실행 방법, 필요한 `.env` 항목, 예시 커맨드와 로그/스크린샷, 관련 이슈 링크를 포함합니다.

## 보안
- 비밀정보는 커밋 금지. `.env`와 환경변수 사용. 운영 환경에서는 `VERIFY_SSL=true` 유지.
- 외부 다운로드(이미지/첨부)와 Confluence 접근 권한을 점검하고 수집하세요.
