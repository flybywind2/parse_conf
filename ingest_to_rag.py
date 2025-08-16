"""
ingest_to_rag.py
시민 개발자용 간단 가이드

이 스크립트는 Confluence 페이지를 가져와 RAG 인덱스로 업로드합니다.
주요 단계:
1) Confluence 페이지 HTML(storage)를 가져옴(fetch_page)
2) 첨부/상대 경로를 절대 URL로 바꾸고 이미지 태그 정리(preprocess_storage_html)
3) HTML→Markdown 변환(markdownify) 및 (옵션) 비전 LLM으로 이미지 설명 추가
4) 업로드 페이로드 생성(build_payload) 후 RAG API로 전송(upload_page_to_rag)
5) 재귀 옵션이 켜져 있으면 하위 페이지까지 순회(traverse_pages)

환경변수(.env)에서 주로 조정하는 값:
- Confluence: CONF_BASE_URL, CONF_USER, CONF_TOKEN, CONF_PAGE_IDS, CONF_RECURSIVE
- RAG: RAG_INSERT_URL, RAG_INDEX_NAME, RAG_API_KEY, PASS_KEY
- 청킹: USE_CHUNK_FACTOR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATOR
- 비전: VISION_ENABLE, VISION_MODEL, VISION_MAX_IMAGES_PER_PAGE

빠른 실행: `python ingest_to_rag.py 12345,67890` 또는 `.env`의 CONF_PAGE_IDS 사용
"""

# ingest_to_rag.py
import os
import json
import time
import uuid
import base64
import hashlib
import mimetypes
import requests
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

from dotenv import load_dotenv
from markdownify import markdownify as md
from bs4 import BeautifulSoup

# Vision LLM
from io import BytesIO
from PIL import Image  # pip install pillow
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()  # .env 파일을 로드하여 인증/설정을 읽습니다.

# ================= 환경 설정 =================
# Confluence
CONF_BASE_URL = os.getenv("CONF_BASE_URL", "").rstrip("/")
CONF_USER     = os.getenv("CONF_USER", "")
CONF_TOKEN    = os.getenv("CONF_TOKEN", "")
CONF_PAGE_IDS = [s.strip() for s in os.getenv("CONF_PAGE_IDS", "").split(",") if s.strip()]

# 재귀 옵션
CONF_RECURSIVE = os.getenv("CONF_RECURSIVE", "false").lower() == "true"
CONF_MAX_DEPTH = int(os.getenv("CONF_MAX_DEPTH", "3"))

# RAG
RAG_INSERT_URL = os.getenv("RAG_INSERT_URL", "http://apigw.sever.net:8000/v2/insert-doc")
RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME", "rp-rag")
RAG_API_KEY    = os.getenv("RAG_API_KEY", "YOUR_RAG_KEY")
PASS_KEY       = os.getenv("PASS_KEY", "YOUR_PASS_KEY")
DEFAULT_PERMISSION_GROUPS = [s.strip() for s in os.getenv("DEFAULT_PERMISSION_GROUPS", "ds").split(",") if s.strip()]

# 청킹 옵션
USE_CHUNK_FACTOR  = os.getenv("USE_CHUNK_FACTOR", "true").lower() == "true"
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_SEPARATOR   = os.getenv("CHUNK_SEPARATOR", " ")

# 네트워크/제어
TIMEOUT_SEC       = 60
MAX_RETRIES       = 3
RETRY_BACKOFF_SEC = 2
PAGE_LIMIT        = 200
VERIFY_SSL        = os.getenv("VERIFY_SSL", "true").lower() == "true"

# Vision LLM (이미지에 대한 요약/설명 생성 — 선택적)
VISION_ENABLE = os.getenv("VISION_ENABLE", "false").lower() == "true"
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.2"))
VISION_MAX_IMAGES_PER_PAGE = int(os.getenv("VISION_MAX_IMAGES_PER_PAGE", "8"))
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "90"))

# Gateway(OpenAI 호환) – 비전/텍스트 공통으로 사용 가능(사내 프록시/게이트웨이)
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL")     # 예: http://apigw.server.net/openai/v1
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")      # 게이트웨이/프록시 키
X_DEP_TICKET      = os.getenv("X_DEP_TICKET", "")
SYSTEM_NAME       = os.getenv("SYSTEM_NAME", "System-Name.")
USER_ID           = os.getenv("USER_ID", "ID")
USER_TYPE         = os.getenv("USER_TYPE", "Type")

# 큰 이미지 축소 파라미터(게이트웨이 500 회피)
VISION_MAX_BYTES_BEFORE = int(os.getenv("VISION_MAX_BYTES_BEFORE", str(3_000_000)))
VISION_TARGET_MAX_BYTES = int(os.getenv("VISION_TARGET_MAX_BYTES", str(1_200_000)))
VISION_MAX_EDGE         = int(os.getenv("VISION_MAX_EDGE", "1600"))
VISION_RETRIES          = int(os.getenv("VISION_RETRIES", "2"))
VISION_RETRY_WAIT       = float(os.getenv("VISION_RETRY_WAIT", "1.5"))
# ======================================================

@dataclass
class CPage:
    page_id: str
    title: str
    space_key: str
    space_name: str
    created_time_iso: str
    updated_time_iso: str
    version: int
    labels: List[str]
    html_storage: str
    markdown: str
    url: str
    depth: int = 0  # 루트=0, 자식=1 ...
    parent_id: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)

def kst_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat()

def _headers_json() -> Dict[str, str]:
    """일반 JSON 호출용 헤더 (Confluence API)."""
    return {"Accept": "application/json", "Content-Type": "application/json"}

def _rag_headers() -> Dict[str, str]:
    """RAG 업로드용 헤더."""
    return {"Content-Type": "application/json", "x-dep-ticket": PASS_KEY, "api-key": RAG_API_KEY}

def _auth() -> Tuple[str, str]:
    """Confluence Basic Auth 튜플. 미설정 시 사용자 친화적 에러."""
    if not (CONF_USER and CONF_TOKEN):
        raise RuntimeError("Confluence 인증 정보가 없습니다. CONF_USER / CONF_TOKEN을 설정하세요.")
    return (CONF_USER, CONF_TOKEN)

# ---------- 공통 HTTP ----------
def _retry_post(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> requests.Response:
    """RAG 업로드 등 POST (Confluence가 아닌 곳). 딜레이는 넣지 않음."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = requests.post(
                url, headers=headers, data=json.dumps(payload),
                timeout=TIMEOUT_SEC, verify=VERIFY_SSL
            )
            if 200 <= resp.status_code < 300:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                print(f"[경고] POST {url} -> {resp.status_code} (재시도 {attempt}/{MAX_RETRIES}) {resp.text[:200]}")
            else:
                return resp
        except requests.RequestException as e:
            last_exc = e
            print(f"[예외] POST {url} 실패: {e} (재시도 {attempt}/{MAX_RETRIES})")
        time.sleep(RETRY_BACKOFF_SEC * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError("요청 실패(원인 불명)")

def _retry_get(url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    """Confluence GET 전용. 매 호출 전에 3초 대기(속도 제한 회피용)."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            time.sleep(3)  # ✅ Confluence API 호출 전 3초 대기
            resp = requests.get(
                url, headers=_headers_json(), params=params,
                auth=_auth(), timeout=TIMEOUT_SEC, verify=VERIFY_SSL, allow_redirects=True
            )
            if 200 <= resp.status_code < 300:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                print(f"[경고] GET {url} -> {resp.status_code} (재시도 {attempt}/{MAX_RETRIES}) {resp.text[:200]}")
            else:
                return resp
        except requests.RequestException as e:
            last_exc = e
            print(f"[예외] GET {url} 실패: {e} (재시도 {attempt}/{MAX_RETRIES})")
        time.sleep(RETRY_BACKOFF_SEC * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError("요청 실패(원인 불명)")

# ---------- Confluence 도우미 ----------
def _absolute_url(rel: str) -> str:
    if not rel:
        return rel
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    return urljoin(CONF_BASE_URL + "/", rel.lstrip("/"))

def _add_download_param(url: str) -> str:
    try:
        u = urlparse(url)
        q = dict(parse_qsl(u.query, keep_blank_values=True))
        q.setdefault("download", "1")
        return urlunparse(u._replace(query=urlencode(q)))
    except Exception:
        return url

def _attachment_download_url(page_id: str, attachment_id: str) -> str:
    # REST 다운로드(권한/인증 문제 회피에 유리)
    return urljoin(
        CONF_BASE_URL + "/",
        f"rest/api/content/{page_id}/child/attachment/{attachment_id}/download"
    )

def fetch_attachments(page_id: str) -> Dict[str, Dict[str, Any]]:
    endpoint = urljoin(CONF_BASE_URL + "/", f"rest/api/content/{page_id}/child/attachment")
    start = 0
    limit = 100
    attachments: Dict[str, Dict[str, Any]] = {}
    while True:
        params = {"limit": limit, "start": start, "expand": "version,metadata"}
        resp = _retry_get(endpoint, params=params)
        data = resp.json() or {}
        results = data.get("results", [])
        for att in results:
            title = att.get("title") or ""
            att_id = att.get("id")
            _links = att.get("_links", {}) or {}
            download_std = _absolute_url(_links.get("download", ""))  # /download/attachments/...
            download_rest = _attachment_download_url(page_id, att_id) if att_id else None
            if title:
                attachments[title] = {
                    "download": _add_download_param(download_rest or download_std),
                    "download_std": _add_download_param(download_std),
                    "download_rest": download_rest and _add_download_param(download_rest),
                    "version": (att.get("version") or {}).get("number"),
                    "mediaType": (att.get("metadata") or {}).get("mediaType") or att.get("type"),
                    "id": att_id,
                }
        if data.get("_links", {}).get("next"):
            start += limit
        else:
            break
    return attachments

def preprocess_storage_html(html: str, page_id: str) -> str:
    """Confluence storage XHTML을 파싱하여 이미지/링크를 절대 URL로 정리합니다."""
    if not html:
        return html
    soup = BeautifulSoup(html, "html.parser")
    attachment_map = fetch_attachments(page_id)

    # <ac:image><ri:attachment ri:filename="..."/></ac:image>
    for ac_img in soup.find_all("ac:image"):
        alt_text = ac_img.get("ac:alt", "") or ac_img.get("alt", "")
        width = ac_img.get("ac:width") or ac_img.get("width")
        height = ac_img.get("ac:height") or ac_img.get("height")
        img_url = None

        ri = ac_img.find("ri:attachment")
        if ri and ri.has_attr("ri:filename"):
            filename = ri["ri:filename"]
            meta = attachment_map.get(filename)
            if meta and meta.get("download"):
                img_url = meta["download"]

        new_img = soup.new_tag("img")
        if img_url:
            new_img["src"] = img_url
        if alt_text:
            new_img["alt"] = alt_text
        if width:
            new_img["width"] = width
        if height:
            new_img["height"] = height
        ac_img.replace_with(new_img)

    # <img src="/download/..."> → 절대 URL
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if src:
            tag["src"] = _absolute_url(src)

    # <a href="/..."> → 절대 URL
    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            a["href"] = _absolute_url(href)

    return str(soup)

# ---------- Vision LLM ----------
def _guess_mime_from_url(url: str) -> str:
    mime, _ = mimetypes.guess_type(url)
    return mime or "image/jpeg"

def _load_image_bytes(src: str) -> Optional[bytes]:
    """이미지 URL/로컬 경로에서 실제 image/* 컨텐츠만 반환. HTML 응답이면 None.
    SSO/리다이렉트로 HTML을 받는 경우를 로그로 안내합니다.
    """
    try:
        headers = {
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "User-Agent": "ingest-bot/1.0",
        }
        if os.path.exists(src):
            with open(src, "rb") as f:
                return f.read()

        if src.startswith("http://") or src.startswith("https://"):
            resp = requests.get(
                src, headers=headers, auth=_auth(),
                timeout=TIMEOUT_SEC, verify=VERIFY_SSL, allow_redirects=True
            )
            if resp.status_code != 200:
                print(f"[경고] 이미지 요청 실패 {src} -> {resp.status_code}")
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" in ctype.lower() or (resp.text or "").lstrip().startswith("<!DOCTYPE html"):
                    print("[힌트] HTML 응답(로그인/권한/SSO 리다이렉트 가능):", (resp.text or "")[:200])
                return None
            ctype = resp.headers.get("Content-Type", "").lower()
            if not ctype.startswith("image/"):
                head = resp.content[:200]
                try:
                    head_str = head.decode("utf-8", errors="ignore")
                except Exception:
                    head_str = str(head)
                print(f"[경고] 비-이미지 응답(Content-Type={ctype}) {src} -> 앞부분: {head_str}")
                return None
            return resp.content
        return None
    except Exception as e:
        print(f"[경고] 이미지 로드 실패 {src}: {e}")
        return None

def _shrink_image_bytes(image_bytes: bytes, target_max_bytes: int, max_edge: int = 1600) -> bytes:
    """큰 이미지를 JPEG로 재인코딩하여 payload 축소. 실패 시 원본 반환.
    게이트웨이/프록시 최대 바이트 한계를 피하기 위한 방어 로직입니다.
    """
    try:
        im = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        scale = min(1.0, max_edge / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        for q in (85, 75, 65, 55, 45, 35, 25):
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=q, optimize=True)
            out = buf.getvalue()
            if len(out) <= target_max_bytes:
                return out
        return out
    except Exception:
        return image_bytes

def _vision_client() -> Optional[OpenAI]:
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

def describe_image_with_llm(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """게이트웨이 직접 호출 → 실패 시 ChatOpenAI 폴백.
    VISION_ENABLE=true일 때만 동작하며, 이미지에 대한 짧은 설명을 생성합니다.
    """
    if not VISION_ENABLE:
        return ""
    # 과대 payload 축소
    if len(image_bytes) > VISION_MAX_BYTES_BEFORE:
        image_bytes = _shrink_image_bytes(image_bytes, VISION_TARGET_MAX_BYTES, max_edge=VISION_MAX_EDGE)
        mime = "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    user_content = [
        {"type": "text", "text": (
            "이미지를 한국어로 자세히 설명하고, 텍스트가 있으면 간략히 추출해 주세요.\n"
            "형식: 1) 한줄 요약 2) 주요 요소 목록 3) 식별된 텍스트(있으면)"
        )},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    ]

    client = _vision_client()
    if client:
        for attempt in range(1, VISION_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=VISION_MODEL,
                    temperature=VISION_TEMPERATURE,
                    messages=[
                        {"role": "system", "content": "당신은 한국어로 간결하고 정확하게 설명하는 비전 비서입니다."},
                        {"role": "user", "content": user_content},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt >= VISION_RETRIES:
                    print(f"[경고] 게이트웨이 비전 호출 실패 최종: {e}")
                time.sleep(VISION_RETRY_WAIT)

    # 폴백: 공식 OpenAI 경유
    try:
        llm = ChatOpenAI(model=VISION_MODEL, temperature=VISION_TEMPERATURE, timeout=VISION_TIMEOUT)
        res = llm.invoke([HumanMessage(content=user_content)])
        return (res.content or "").strip()
    except Exception as e:
        return f"(이미지 설명 실패: {e})"

def build_image_descriptions_block(html_with_abs_img: str) -> Tuple[str, List[str]]:
    """HTML 내 <img>들을 스캔하여 일부 이미지를 비전 LLM으로 설명합니다.
    반환: (마크다운 블록 문자열, 처리된 이미지 URL 목록)
    """
    if not VISION_ENABLE:
        return "", []
    soup = BeautifulSoup(html_with_abs_img, "html.parser")
    imgs = soup.find_all("img")
    if not imgs:
        return "", []
    lines = ["\n\n## 이미지 설명\n"]
    count = 0
    processed_urls: List[str] = []
    for idx, tag in enumerate(imgs, 1):
        if count >= VISION_MAX_IMAGES_PER_PAGE:
            lines.append(f"- (나머지 {len(imgs) - count}개 이미지는 생략)")
            break
        src = tag.get("src") or ""
        if not src:
            continue
        img_bytes = _load_image_bytes(src)
        if not img_bytes:
            lines.append(f"- [{idx}] {src}\n  - 설명: (이미지를 불러오지 못했습니다)")
            continue
        mime = _guess_mime_from_url(src)
        desc = describe_image_with_llm(img_bytes, mime=mime)
        alt_txt = (tag.get("alt") or "").strip()
        block = [f"- [{idx}] {src}"]
        if alt_txt:
            block.append(f"  - alt: {alt_txt}")
        block.append(f"  - 설명: {desc}")
        lines.append("\n".join(block))
        processed_urls.append(src)
        count += 1
    return "\n".join(lines), processed_urls

# ---------- Confluence API ----------
def fetch_page(page_id: str, depth: int = 0, parent_id: Optional[str] = None) -> CPage:
    """단일 Confluence 페이지를 조회하고 CPage 구조로 변환."""
    if not CONF_BASE_URL:
        raise RuntimeError("CONF_BASE_URL이 비어 있습니다.")
    endpoint = urljoin(CONF_BASE_URL + "/", f"rest/api/content/{page_id}")
    params = {"expand": "body.storage,history,version,space,metadata.labels"}
    resp = _retry_get(endpoint, params=params)
    data = resp.json()

    title = data.get("title", f"page-{page_id}")
    space = data.get("space") or {}
    space_key = space.get("key", "")
    space_name = space.get("name", "")
    history = data.get("history") or {}
    created_iso = history.get("createdDate") or kst_now_iso()

    version = data.get("version") or {}
    number = int(version.get("number") or 1)
    when = version.get("when") or created_iso
    updated_iso = when

    labels_data = (data.get("metadata") or {}).get("labels") or {}
    label_results = labels_data.get("results") or []
    labels = [it.get("name") for it in label_results if isinstance(it, dict) and it.get("name")]

    body = (data.get("body") or {}).get("storage") or {}
    raw_html = body.get("value", "")

    fixed_html = preprocess_storage_html(raw_html, str(page_id))
    markdown = md(fixed_html, heading_style="ATX", bullets="*")

    img_block, vision_urls = build_image_descriptions_block(fixed_html)
    if img_block.strip():
        markdown = markdown + "\n" + img_block

    page_url = urljoin(CONF_BASE_URL + "/", f"pages/viewpage.action?pageId={page_id}")

    return CPage(
        page_id=str(page_id),
        title=title,
        space_key=space_key,
        space_name=space_name,
        created_time_iso=created_iso,
        updated_time_iso=updated_iso,
        version=number,
        labels=labels,
        html_storage=fixed_html,
        markdown=markdown,
        url=page_url,
        depth=depth,
        parent_id=parent_id,
        image_urls=vision_urls
    )

def fetch_children_page_ids(parent_id: str) -> List[str]:
    """부모 페이지의 하위 페이지 ID 목록 가져오기(페이지네이션 처리)."""
    endpoint = urljoin(CONF_BASE_URL + "/", f"rest/api/content/{parent_id}/child/page")
    start = 0
    ids: List[str] = []
    while True:
        params = {"limit": PAGE_LIMIT, "start": start, "expand": "version"}
        resp = _retry_get(endpoint, params=params)
        data = resp.json() or {}
        results = data.get("results", [])
        for it in results:
            pid = it.get("id")
            if pid:
                ids.append(str(pid))
        if data.get("_links", {}).get("next"):
            start += PAGE_LIMIT
        else:
            break
    return ids

def traverse_pages(root_ids: List[str], recursive: bool, max_depth: int) -> List[CPage]:
    """루트 페이지 집합을 시작으로 DFS 순회하여 페이지 목록을 반환."""
    pages: List[CPage] = []
    visited: Set[str] = set()

    def dfs(current_id: str, depth: int, parent: Optional[str]):
        if current_id in visited:
            return
        visited.add(current_id)
        try:
            page = fetch_page(current_id, depth=depth, parent_id=parent)
            pages.append(page)
            print(f"[가져옴] depth={depth} id={current_id} '{page.title}' (v{page.version}) labels={page.labels}")
        except Exception as e:
            print(f"[경고] 페이지 가져오기 실패 id={current_id}: {e}")
            return

        if recursive and depth < max_depth:
            try:
                child_ids = fetch_children_page_ids(current_id)
                if child_ids:
                    print(f"  └─ 하위 페이지 {len(child_ids)}건")
                for cid in child_ids:
                    dfs(cid, depth + 1, current_id)
            except Exception as e:
                print(f"[경고] 하위 페이지 조회 실패 id={current_id}: {e}")

    for rid in root_ids:
        dfs(rid, 0, None)

    return pages

# ---------- 업로드 ----------
def make_doc_id_from_conf(page: CPage) -> str:
    """Confluence 페이지 ID+버전으로부터 안정적인 해시형 문서 ID 생성."""
    base = f"{page.page_id}::v{page.version}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest().upper()[:16]

def build_payload(page: CPage) -> Dict[str, Any]:
    """RAG 인서트 API 스키마에 맞춘 데이터 페이로드 구성.
    비전이 활성화되고 이미지가 처리된 경우 vision_image_urls를 포함합니다.
    """
    data = {
        "doc_id": make_doc_id_from_conf(page),
        "title": page.title,
        "content": page.markdown,
        "permission_groups": DEFAULT_PERMISSION_GROUPS,
        "created_time": page.created_time_iso or kst_now_iso(),
        "additionalField": "confluence_page",
        "source": page.url,
        "conf_page_id": page.page_id,
        "conf_space_key": page.space_key,
        "conf_space_name": page.space_name,
        "conf_version": page.version,
        "conf_labels": page.labels,
        "conf_updated_time": page.updated_time_iso,
        "conf_depth": page.depth,
        "conf_parent_id": page.parent_id,
    }
    # Vision LLM이 활성화되어 있고 처리한 이미지가 있다면 URL 목록을 포함
    if VISION_ENABLE and page.image_urls:
        data["vision_image_urls"] = page.image_urls
    payload = {"index_name": RAG_INDEX_NAME, "data": data}
    if USE_CHUNK_FACTOR:
        payload["chunk_factor"] = {
            "logic": "fixed_size",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "separator": CHUNK_SEPARATOR
        }
    return payload

def upload_page_to_rag(page: CPage) -> None:
    """단일 페이지를 RAG 인덱스로 업로드하고 응답을 로그로 출력."""
    payload = build_payload(page)
    print(f"\n[업로드] pageId={page.page_id} depth={page.depth} title='{page.title}' -> {RAG_INSERT_URL}")
    resp = _retry_post(RAG_INSERT_URL, _rag_headers(), payload)
    print(f"[응답] {resp.status_code}")
    try:
        print(resp.json())
    except Exception:
        print(resp.text[:400])

# ---------- 메인 ----------
def main():
    """명령행 진입점: 입력 pageId(또는 .env)→순회→업로드."""
    import sys
    if len(sys.argv) > 1:
        page_ids = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    else:
        page_ids = CONF_PAGE_IDS

    if not page_ids:
        print("[오류] pageId가 없습니다. .env의 CONF_PAGE_IDS 또는 명령행 인자로 지정하세요.")
        print("예) python ingest_to_rag.py 12345,67890")
        return

    print(f"[입력] pageIds: {', '.join(page_ids)}  recursive={CONF_RECURSIVE}  max_depth={CONF_MAX_DEPTH}  vision={VISION_ENABLE}")

    pages = traverse_pages(page_ids, recursive=CONF_RECURSIVE, max_depth=CONF_MAX_DEPTH)
    if not pages:
        print("[오류] 가져온 페이지가 없습니다.")
        return

    for page in pages:
        try:
            upload_page_to_rag(page)
        except Exception as e:
            print(f"[경고] 업로드 실패 pageId={page.page_id}: {e}")

if __name__ == "__main__":
    main()
