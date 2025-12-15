# src/input_handling.py
import re
import os
import json
import hashlib
from typing import Tuple, Dict, List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from pydantic import HttpUrl

# attempt language id backends in order: pycld3 -> fasttext -> langdetect
try:
    import pycld3 as cld3
except Exception:
    cld3 = None

try:
    import fasttext
    _fasttext_model = None
except Exception:
    fasttext = None
    _fasttext_model = None

try:
    from langdetect import detect_langs
except Exception:
    detect_langs = None

from src.schema import InputRecord, SourceMetadata


# ---------- Utilities ----------
def _sha1_of_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def _slug_from_title(title: Optional[str], fallback: str) -> str:
    if title:
        s = re.sub(r"[^0-9a-zA-Z]+", "-", title).strip("-").lower()
        if s:
            return s[:60]
    return fallback


# ---------- Source detection ----------
def detect_source_type(path_or_text: str, maybe_url: Optional[str] = None) -> str:
    """
    Heuristic: return one of 'gutenberg', 'html', 'json_api', 'txt', 'unknown'
    """
    # if looks like a URL:
    if maybe_url:
        parsed = urlparse(maybe_url)
        if parsed.netloc and "gutenberg" in parsed.netloc:
            return "gutenberg"
    # If this is a file path or a content string: quick heuristics
    if os.path.exists(path_or_text):
        _, ext = os.path.splitext(path_or_text.lower())
        if ext in (".html", ".htm"):
            return "html"
        if ext in (".json",):
            return "json_api"
        if ext in (".txt",):
            # check for Gutenberg metadata markers
            with open(path_or_text, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
            if "Project Gutenberg" in head or "GUTENBERG" in head.upper():
                return "gutenberg"
            # otherwise plain text
            return "txt"
    else:
        # content snippet heuristics
        s = path_or_text.strip()[:200].lower()
        if s.startswith("<!doctype") or s.startswith("<html") or "<body" in s:
            return "html"
        if s.startswith("{") or s.startswith("["):
            return "json_api"
        if "project gutenberg" in s or "gutenberg" in s:
            return "gutenberg"
    return "unknown"


# ---------- Parsers ----------
def parse_gutenberg_text(text: str) -> Tuple[SourceMetadata, List[str]]:
    """
    Extract title, author if present and paragraphs. Gutenberg files vary widely,
    but they commonly have a header and footer markers like '*** START OF THIS PROJECT GUTENBERG EBOOK'
    """
    meta = {}
    # find header block
    header_matches = list(re.finditer(r'((project|proyecto|projekt|projet) gutenberg|Gutenberg Projekt).*?$', header_region, re.IGNORECASE | re.MULTILINE))
    footer_match = re.search(r'^.*(?:end of (?:the|this)?\s*project gutenberg|fin (?:del|de texto)\s*proyecto gutenberg|Ende dieses Projekt Gutenberg).*$', text, re.IGNORECASE | re.MULTILINE)

    body = text
    if header_matches:
		last_header = header_matches[-1]
		body = text[last_header.end():] 
        header = text[:last_header.start()]
        # naive title/author extraction from header
        title_match = re.search(r"Title:\s*(.+)", header)
        author_match = re.search(r"Author:\s*(.+)", header)
        if title_match:
            meta["title"] = title_match.group(1).strip()
        if author_match:
            meta["authors"] = [a.strip() for a in author_match.group(1).split(";") if a.strip()]
	if footer_match:
		# Remove everything from the start of that line onward
		body = body[:footer_match.start()]

	# Replace single newlines within paragraphs with space
	body = re.sub(r'(?<!\n)\n(?!\n)', ' ', body)
	# Strip 
	body = re.sub(r'_(.*?)_', r'\1', body)
	# Collapse multiple spaces
	body = re.sub(r' +', ' ', body)
	body = re.sub(r'\n{3,}', '\n\n', body)

	body = "\n".join(
		line for line in body.splitlines()
		if not re.search(r'(project gutenberg|www.gutenberg.org|gutenberg2000.de|gutenberg.spiegel.de)', line, re.IGNORECASE)
	)
    paragraphs = _split_paragraphs(body)
    md = SourceMetadata(
        title=meta.get("title"),
        authors=meta.get("authors"),
        source_type="gutenberg",
        raw_meta=meta,
    )
    return md, paragraphs


def parse_html(text: str, base_url: Optional[str] = None) -> Tuple[SourceMetadata, List[str]]:
    soup = BeautifulSoup(text, "html.parser")
    # title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    # attempt to extract meta tags
    meta = {}
    for name in ("author", "dc.creator", "og:title", "og:site_name"):
        m = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if m and m.get("content"):
            meta[name] = m["content"]
    # extract visible text and preserve paragraph blocks
    paras = []
    for p in soup.find_all(["p", "div"]):
        txt = p.get_text(separator=" ", strip=True)
        if txt and len(txt) > 20:  # filter tiny junk
            paras.append(txt)
    if not paras:
        # fallback: large text blocks by splitting body text by double newlines
        body = soup.get_text("\n")
        paras = _split_paragraphs(body)
    md = SourceMetadata(title=title, authors=[meta.get("author")] if meta.get("author") else None, source_type="html",
                        raw_meta=meta)
    if base_url:
        try:
            md.url = base_url  # type: ignore[attr-defined]
        except Exception:
            pass
    return md, paras


def parse_json_api(json_obj: dict) -> Tuple[SourceMetadata, List[str]]:
    """
    Expecting standard patterns:
    - {'title':..., 'content':...}
    - Or a feed with entries
    """
    meta = {}
    paragraphs = []
    if "title" in json_obj:
        meta["title"] = json_obj["title"]
    if "author" in json_obj:
        meta["authors"] = [json_obj["author"]]
    # common content fields
    for key in ("content", "body", "text", "description"):
        if key in json_obj:
            raw = json_obj[key]
            if isinstance(raw, str):
                paragraphs = _split_paragraphs(raw)
            elif isinstance(raw, list):
                paragraphs = [p for p in raw if isinstance(p, str)]
            break
    # feed-like
    if not paragraphs and "entries" in json_obj and isinstance(json_obj["entries"], list):
        for e in json_obj["entries"]:
            if isinstance(e, dict) and "content" in e:
                paragraphs.extend(_split_paragraphs(e["content"]))
    md = SourceMetadata(title=meta.get("title"), authors=meta.get("authors"), source_type="json_api", raw_meta=json_obj)
    return md, paragraphs


def parse_plain_text(text: str) -> Tuple[SourceMetadata, List[str]]:
    paras = _split_paragraphs(text)
    md = SourceMetadata(source_type="txt")
    return md, paras


def _split_paragraphs(text: str) -> List[str]:
    # split on double newlines or lines that look like paragraph separators
    # keep paragraphs trimmed, ignore very short lines (unless they are the only content)
    blocks = re.split(r"\n{2,}|\r\n{2,}", text)
    paras = [re.sub(r"\s+", " ", b).strip() for b in blocks if b and len(b.strip()) > 0]
    # but if nothing longer than 20 chars, keep shorter ones
    if not any(len(p) > 20 for p in paras) and paras:
        return paras
    return [p for p in paras if len(p) > 20]


# ---------- Language Identification ----------
def identify_language(text: str, prefer_top_k: int = 1) -> Dict[str, float]:
    """
    Return dict: {lang_code: score}
    Use pycld3 first (probabilities), then fasttext if available, else langdetect.
    """
    short = text.strip()[:3000]  # clamp length
    if cld3:
        try:
            r = cld3.get_language(short)
            if r and r.language:
                return {r.language: float(r.probability)}
        except Exception:
            pass

    if fasttext:
        global _fasttext_model
        try:
            if _fasttext_model is None:
                # you must download model 'lid.176.ftz' or 'lid.176.bin' in production and set path
                # here we try to load a packaged, user-provided path, else raise
                raise RuntimeError("fastText model not configured")
            preds = _fasttext_model.predict(short, k=prefer_top_k)
            labels, probs = preds
            out = {}
            for lab, p in zip(labels, probs):
                lang = lab.replace("__label__", "")
                out[lang] = float(p)
            return out
        except Exception:
            pass

    if detect_langs:
        try:
            ds = detect_langs(short)
            out = {}
            for d in ds[:prefer_top_k]:
                out[d.lang] = float(d.prob)
            return out
        except Exception:
            pass

    # fallback - unknown
    return {"und": 0.0}


# ---------- Main API ----------
def ingest_from_path(path: str, doc_id: Optional[str] = None, base_url: Optional[str] = None) -> InputRecord:
    with open(path, "rb") as f:
        b = f.read()
    text = b.decode("utf-8", errors="ignore")
    return ingest_from_text(text, source_name=path, doc_id=doc_id, base_url=base_url)


def ingest_from_text(text: str, source_name: Optional[str] = None, doc_id: Optional[str] = None,
                     base_url: Optional[str] = None, provided_source_type: Optional[str] = None) -> InputRecord:
    # determine id
    raw_bytes = (text or "").encode("utf-8")
    fallback_id = _sha1_of_bytes(raw_bytes)[:12]
    if doc_id is None:
        doc_id = f"{_slug_from_title(None, fallback_id)}"

    # detect source type
    source_type = provided_source_type or detect_source_type(source_name or text, maybe_url=base_url)

    # parse
    if source_type == "gutenberg":
        meta, paras = parse_gutenberg_text(text)
    elif source_type == "html":
        meta, paras = parse_html(text, base_url=base_url)
    elif source_type == "json_api":
        try:
            js = json.loads(text)
        except Exception:
            js = {}
        meta, paras = parse_json_api(js)
    elif source_type == "txt":
        meta, paras = parse_plain_text(text)
    else:
        # fallback: try html parse, then plain text
        meta, paras = parse_html(text, base_url=base_url)
        if not paras:
            meta, paras = parse_plain_text(text)

    # language detection (on whole document)
    lang_scores = identify_language(" ".join(paras) if paras else text)
    # pick top
    top_lang = max(lang_scores.items(), key=lambda kv: kv[1])[0] if lang_scores else None
    meta.language = top_lang  # type: ignore[attr-defined]
    meta.source_type = source_type

    # compute paragraph_orig_indices (simple identity mapping)
    paragraph_orig_indices = list(range(len(paras)))

    record = InputRecord(
        id=doc_id,
        metadata=meta,
        paragraphs=paras,
        paragraph_orig_indices=paragraph_orig_indices,
        provenance={
            "source_name": source_name,
            "base_url": base_url,
            "detected_source_type": source_type,
            "lang_scores": lang_scores,
        }
    )
    return record


# ---------- Small CLI utility ----------
def ingest_and_save(path_or_text: str, out_path: str, is_file: bool = True, base_url: Optional[str] = None):
    if is_file:
        rec = ingest_from_path(path_or_text, base_url=base_url)
    else:
        rec = ingest_from_text(path_or_text, source_name=None, base_url=base_url)
    # write to JSON
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rec.json(indent=2, ensure_ascii=False))
    return out_path
