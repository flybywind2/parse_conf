# Repository Guidelines

## Project Structure & Module Organization
- `ask_rag_llm.py`: Retrieves documents from a RAG endpoint, composes a context block, and queries an LLM. Prints answer and brief source metadata.
- `ingest_to_rag.py`: Crawls Confluence pages (optionally recursive), normalizes content (HTML → Markdown, absolute links, optional image descriptions), and uploads to the RAG index.
- `.env` (create locally): Stores runtime configuration and API keys used by both scripts.

## Build, Test, and Development Commands
- Setup venv (Windows): `python -m venv .venv && .venv\\Scripts\\activate`
- Install deps: `pip install -U pip && pip install python-dotenv requests openai langchain-openai markdownify beautifulsoup4 pillow`
- Run ingest (IDs comma-separated): `python ingest_to_rag.py 12345,67890`
- Run Q&A with RAG: `python ask_rag_llm.py`

## Configuration Tips
Create a `.env` in the repo root. Minimal example:
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

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; modules and functions `snake_case`, classes `CamelCase`.
- Keep functions focused; reuse helpers; prefer explicit env reads at top of files.
- Follow existing print/log style and Korean user-facing messages.

## Testing Guidelines
- No test suite present. Validate by running with a small `CONF_PAGE_IDS` and confirming upload logs, then query with `ask_rag_llm.py`.
- If adding tests, use `pytest`, files named `tests/test_*.py`, and aim for fast, isolated unit tests around helpers.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits recommended (e.g., `feat: add image OCR`, `fix: retry on 502`). Keep messages in imperative mood and scoped changesets.
- PRs: include purpose, how to run, relevant `.env` entries, sample commands, and logs/screenshots. Link issues when applicable.

## Security
- Never commit secrets; use `.env` and environment variables. Keep `VERIFY_SSL=true` for production. Review external downloads (images/attachments) and Confluence permissions before ingesting.
