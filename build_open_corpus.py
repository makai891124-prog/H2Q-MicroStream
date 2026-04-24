"""
build_open_corpus.py
====================
Build a mixed open corpus for H2Q local structure learning.

Sources:
1) arXiv latest math/cs.AI metadata + abstracts (Atom API)
2) arXiv PDF full-text extraction stream (configurable pages per paper)
3) Hugging Face dataset text cards (lightweight endpoint)
4) GitHub repository README and selected source files from public repos

Output:
- data/open_corpus/open_corpus.txt
- data/open_corpus/open_corpus.bin
- data/open_corpus/source_manifest.json

This script intentionally keeps dependencies minimal (requests only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List

import requests

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

logging.getLogger("pypdf").setLevel(logging.ERROR)


ARXIV_API = "https://export.arxiv.org/api/query"
HF_DATASET_INDEX = "https://huggingface.co/api/datasets"
GITHUB_API = "https://api.github.com"
DEFAULT_HEADERS = {"User-Agent": "H2Q-MicroStream/1.0 (research-bot)"}


@dataclass
class SourceItem:
    source_type: str
    title: str
    url: str
    content_sha256: str
    content_bytes: int


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_arxiv_math_and_ai(max_results: int, timeout: int) -> List[tuple[SourceItem, str]]:
    query = "cat:math OR cat:cs.AI"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=timeout)
    r.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)

    out: List[tuple[SourceItem, str]] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

        text = clean_text(
            f"[ARXIV]\nTitle: {title}\nPublished: {published}\nURL: {entry_id}\nAbstract:\n{summary}\n"
        )
        item = SourceItem(
            source_type="arxiv",
            title=title,
            url=entry_id,
            content_sha256=sha256_text(text),
            content_bytes=len(text.encode("utf-8")),
        )
        out.append((item, text))

    return out


def fetch_arxiv_entries(max_results: int, timeout: int) -> List[dict]:
    query = "cat:math OR cat:cs.AI"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
    r.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    entries: List[dict] = []

    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            t = (link.attrib.get("type") or "").lower()
            if t == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break

        if not pdf_url and entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"

        entries.append(
            {
                "title": title,
                "summary": summary,
                "entry_id": entry_id,
                "published": published,
                "pdf_url": pdf_url,
            }
        )

    return entries


def normalize_arxiv_pdf_urls(pdf_url: str) -> List[str]:
    if not pdf_url:
        return []

    urls = []
    u = pdf_url.strip()
    if u.startswith("http://"):
        u = "https://" + u[len("http://") :]

    if not u.endswith(".pdf"):
        u = u + ".pdf"

    urls.append(u)

    # Mirror fallback: arxiv.org <-> export.arxiv.org
    if "arxiv.org/" in u and "export.arxiv.org/" not in u:
        urls.append(u.replace("https://arxiv.org/", "https://export.arxiv.org/"))
    if "export.arxiv.org/" in u:
        urls.append(u.replace("https://export.arxiv.org/", "https://arxiv.org/"))

    # Deduplicate while preserving order.
    seen = set()
    dedup = []
    for x in urls:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def fetch_arxiv_pdf_fulltext(
    max_results: int,
    timeout: int,
    out_dir: str,
    max_pages: int,
    max_pdf_bytes: int,
) -> List[tuple[SourceItem, str]]:
    if PdfReader is None:
        print("[corpus][WARN] pypdf not installed, skip arXiv PDF full-text extraction")
        return []

    os.makedirs(out_dir, exist_ok=True)
    entries = fetch_arxiv_entries(max_results=max_results, timeout=timeout)
    out: List[tuple[SourceItem, str]] = []

    for e in entries:
        raw_pdf_url = e.get("pdf_url", "")
        candidate_urls = normalize_arxiv_pdf_urls(raw_pdf_url)
        if not candidate_urls:
            continue

        # Build a stable local file name from arXiv id.
        paper_id = e.get("entry_id", "").split("/")[-1].replace("v", "_v")
        if not paper_id:
            paper_id = sha256_text(pdf_url)[:16]
        pdf_path = os.path.join(out_dir, f"{paper_id}.pdf")

        try:
            ok = False
            final_pdf_url = ""

            # Reuse cached pdf if present.
            if os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0:
                ok = True
                final_pdf_url = candidate_urls[0]

            if not ok:
                for pdf_url in candidate_urls:
                    try:
                        rr = requests.get(
                            pdf_url,
                            timeout=(5, timeout),
                            headers=DEFAULT_HEADERS,
                            stream=True,
                            allow_redirects=True,
                        )
                        if rr.status_code != 200:
                            continue
                        written = 0
                        too_large = False
                        with open(pdf_path, "wb") as f:
                            for chunk in rr.iter_content(chunk_size=1024 * 64):
                                if not chunk:
                                    continue
                                written += len(chunk)
                                if written > max_pdf_bytes:
                                    too_large = True
                                    break
                                f.write(chunk)

                        if too_large:
                            try:
                                os.remove(pdf_path)
                            except Exception:
                                pass
                            continue

                        if os.path.getsize(pdf_path) > 0:
                            ok = True
                            final_pdf_url = pdf_url
                            break
                    except Exception:
                        continue

            if not ok:
                continue

            reader = PdfReader(pdf_path)
            pages_text = []
            max_take = min(max_pages, len(reader.pages))
            for i in range(max_take):
                try:
                    ptxt = reader.pages[i].extract_text() or ""
                except Exception:
                    ptxt = ""
                if ptxt.strip():
                    pages_text.append(ptxt)

            if not pages_text:
                continue

            text = clean_text(
                "[ARXIV_PDF_FULLTEXT]\n"
                f"Title: {e.get('title', '')}\n"
                f"Published: {e.get('published', '')}\n"
                f"URL: {e.get('entry_id', '')}\n"
                f"PDF: {final_pdf_url}\n"
                f"ExtractedPages: {max_take}\n\n"
                + "\n\n".join(pages_text)
            )
            item = SourceItem(
                source_type="arxiv_pdf",
                title=e.get("title", ""),
                url=final_pdf_url,
                content_sha256=sha256_text(text),
                content_bytes=len(text.encode("utf-8", errors="ignore")),
            )
            out.append((item, text))
        except Exception:
            continue

    return out


def fetch_hf_dataset_cards(limit: int, timeout: int) -> List[tuple[SourceItem, str]]:
    params = {"limit": limit, "sort": "downloads", "direction": -1}
    r = requests.get(HF_DATASET_INDEX, params=params, timeout=timeout)
    r.raise_for_status()
    rows = r.json()

    out: List[tuple[SourceItem, str]] = []
    for row in rows:
        ds_id = row.get("id")
        if not ds_id:
            continue

        tags = ", ".join(row.get("tags", []) or [])
        likes = row.get("likes", 0)
        downloads = row.get("downloads", 0)

        text = clean_text(
            "[HUGGINGFACE_DATASET]\\n"
            f"Dataset: {ds_id}\\n"
            f"URL: https://huggingface.co/datasets/{ds_id}\\n"
            f"Likes: {likes}\\n"
            f"Downloads: {downloads}\\n"
            f"Tags: {tags}\\n"
        )
        item = SourceItem(
            source_type="huggingface",
            title=ds_id,
            url=f"https://huggingface.co/datasets/{ds_id}",
            content_sha256=sha256_text(text),
            content_bytes=len(text.encode("utf-8")),
        )
        out.append((item, text))

    return out


def fetch_github_repos(repo_list: List[str], timeout: int) -> List[tuple[SourceItem, str]]:
    out: List[tuple[SourceItem, str]] = []

    for repo in repo_list:
        owner, name = repo.split("/", 1)

        # Repo metadata
        meta_url = f"{GITHUB_API}/repos/{owner}/{name}"
        try:
            meta = requests.get(meta_url, timeout=timeout)
            if meta.status_code != 200:
                continue
            meta_j = meta.json()
        except Exception:
            continue

        # README
        readme_url = f"https://raw.githubusercontent.com/{owner}/{name}/HEAD/README.md"
        readme = ""
        try:
            rr = requests.get(readme_url, timeout=timeout)
            if rr.status_code == 200:
                readme = rr.text
        except Exception:
            pass

        topics = ", ".join(meta_j.get("topics", []) or [])
        desc = meta_j.get("description") or ""
        stars = meta_j.get("stargazers_count", 0)

        text = clean_text(
            f"[GITHUB_REPO]\nRepo: {repo}\nURL: https://github.com/{repo}\nStars: {stars}\nTopics: {topics}\nDescription: {desc}\n\nREADME:\n{readme}\n"
        )
        item = SourceItem(
            source_type="github",
            title=repo,
            url=f"https://github.com/{repo}",
            content_sha256=sha256_text(text),
            content_bytes=len(text.encode("utf-8")),
        )
        out.append((item, text))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build open corpus for H2Q daemon training")
    parser.add_argument("--out-dir", type=str, default="data/open_corpus")
    parser.add_argument("--arxiv-max", type=int, default=120)
    parser.add_argument("--arxiv-pdf-max", type=int, default=40)
    parser.add_argument("--arxiv-pdf-pages", type=int, default=8)
    parser.add_argument("--arxiv-pdf-max-bytes", type=int, default=12582912, help="Max bytes per PDF file")
    parser.add_argument("--hf-max", type=int, default=40)
    parser.add_argument(
        "--github-repos",
        type=str,
        default="karpathy/nanoGPT,huggingface/transformers,pytorch/pytorch,openai/gym,facebookresearch/esm",
        help="Comma separated owner/repo list",
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--target-mb",
        type=int,
        default=256,
        help="Expand merged corpus to at least this many MB by stochastic tiling",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    arxiv_pdf_dir = os.path.join(args.out_dir, "arxiv_pdfs")

    all_items: List[SourceItem] = []
    all_text_parts: List[str] = []

    print("[corpus] Fetching arXiv...")
    try:
        for item, text in fetch_arxiv_math_and_ai(args.arxiv_max, args.timeout):
            all_items.append(item)
            all_text_parts.append(text)
    except Exception as e:
        print(f"[corpus][WARN] arXiv abstract fetch failed: {e}")

    print("[corpus] Fetching arXiv PDF full-text...")
    try:
        for item, text in fetch_arxiv_pdf_fulltext(
            max_results=args.arxiv_pdf_max,
            timeout=args.timeout,
            out_dir=arxiv_pdf_dir,
            max_pages=args.arxiv_pdf_pages,
            max_pdf_bytes=args.arxiv_pdf_max_bytes,
        ):
            all_items.append(item)
            all_text_parts.append(text)
    except Exception as e:
        print(f"[corpus][WARN] arXiv PDF fetch failed: {e}")

    print("[corpus] Fetching Hugging Face dataset cards...")
    try:
        for item, text in fetch_hf_dataset_cards(args.hf_max, args.timeout):
            all_items.append(item)
            all_text_parts.append(text)
    except Exception as e:
        print(f"[corpus][WARN] Hugging Face fetch failed: {e}")

    repo_list = [x.strip() for x in args.github_repos.split(",") if x.strip()]
    print(f"[corpus] Fetching GitHub repos: {len(repo_list)}")
    try:
        for item, text in fetch_github_repos(repo_list, args.timeout):
            all_items.append(item)
            all_text_parts.append(text)
    except Exception as e:
        print(f"[corpus][WARN] GitHub fetch failed: {e}")

    if not all_text_parts:
        raise RuntimeError("No corpus text collected from any source")

    merged_text = clean_text("\n\n".join(all_text_parts))

    target_bytes = max(1, args.target_mb) * 1024 * 1024
    if len(merged_text.encode("utf-8")) < target_bytes:
        random.seed(20260410)
        blocks = [t for t in all_text_parts if t.strip()]
        tiled_parts: List[str] = []
        cur_bytes = 0
        while cur_bytes < target_bytes and blocks:
            random.shuffle(blocks)
            for blk in blocks:
                if cur_bytes >= target_bytes:
                    break
                # Random local span to increase sequence variety.
                if len(blk) > 2000:
                    s = random.randint(0, max(0, len(blk) - 2000))
                    e = min(len(blk), s + random.randint(1200, 4000))
                    seg = blk[s:e]
                else:
                    seg = blk
                tiled_parts.append(seg)
                cur_bytes += len(seg.encode("utf-8", errors="ignore"))
        merged_text = clean_text("\n\n".join(tiled_parts))

    txt_path = os.path.join(args.out_dir, "open_corpus.txt")
    bin_path = os.path.join(args.out_dir, "open_corpus.bin")
    manifest_path = os.path.join(args.out_dir, "source_manifest.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(merged_text)

    with open(bin_path, "wb") as f:
        f.write(merged_text.encode("utf-8", errors="ignore"))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": [asdict(x) for x in all_items],
        "item_count": len(all_items),
        "text_bytes": len(merged_text.encode("utf-8")),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[corpus] Done")
    print(f"[corpus] items={manifest['item_count']}")
    print(f"[corpus] text_bytes={manifest['text_bytes']}")
    print(f"[corpus] txt={txt_path}")
    print(f"[corpus] bin={bin_path}")
    print(f"[corpus] manifest={manifest_path}")


if __name__ == "__main__":
    main()
