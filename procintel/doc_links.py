"""Links from a passage back to the original World Bank document.

doc ids end in the World Bank Documents & Reports id (ICR_P160162_34360386 -> 34360386); the WDS API maps that id to the
official PDF. Two sources, so the app needs no outbound call at request time:
  * the document's own metadata: the fetch stores the PDF URL in the doc sidecar (meta.pdf_url), which flows into bronze_docs
    and the serving snapshot; Platform registers it on load. This is how documents fetched on Databricks get a link;
  * doc_links.json next to this file: a fallback for documents fetched before the URL was kept in their metadata.
Page numbers are PDF pages: the WB text renditions keep one form feed per PDF page (verified on ICR_P160162_34360386: the
form-feed page of a passage = the page pypdf finds it on, 28 of 67), so `pdf_url#page=N` opens the right page.

  python -m procintel.doc_links data/live/docs data/incoming_rw/docs     # (re)build doc_links.json from downloaded docs"""
from __future__ import annotations
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "doc_links.json")
_CACHE: dict | None = None


def wds_id(doc_id: str) -> str | None:
    m = re.search(r"_(\d{6,})$", doc_id)
    return m.group(1) if m else None


def load() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.load(open(PATH, encoding="utf-8")) if os.path.exists(PATH) else {}
    return _CACHE


def pdf_url(doc_id: str) -> str | None:
    return load().get(doc_id)


def page_link(doc_id: str, page: int | None) -> str | None:
    """The official PDF opened at the passage's page (#page=N works in Chrome, Edge, Firefox and Acrobat)."""
    url = pdf_url(doc_id)
    return f"{url}#page={int(page)}" if url and page else url


def register(docs):
    """Take each loaded document's PDF URL from its metadata (in memory only: the package folder is not written, it is
    read-only or a throwaway copy on Databricks). Metadata wins over doc_links.json."""
    links = load()
    for d in docs:
        url = (d.get("meta") or {}).get("pdf_url")
        if url:
            links[d["doc_id"]] = url


def build(folders):
    from .live_wb import get_json, SEARCH
    links = load()
    ids = sorted({os.path.splitext(f)[0] for d in folders for f in os.listdir(d) if f.endswith(".txt")})
    todo = [d for d in ids if d not in links and wds_id(d)]
    print(f"{len(ids)} documents, {len(todo)} to resolve")
    for i, doc_id in enumerate(todo):
        try:
            res = get_json(f"{SEARCH}/v3/wds", dict(format="json", fl="pdfurl,url", id=wds_id(doc_id)))
            docs = [v for v in (res.get("documents") or {}).values() if isinstance(v, dict) and v.get("id")]
            url = (docs[0].get("pdfurl") or docs[0].get("url")) if docs else None
            if url:
                links[doc_id] = url.replace("http://", "https://")
        except Exception as e:
            print("skip", doc_id, e)
        if i % 25 == 0:
            print(i, doc_id, links.get(doc_id))
        time.sleep(0.2)
    json.dump(dict(sorted(links.items())), open(PATH, "w", encoding="utf-8"), indent=0)
    print(f"{len(links)} links -> {PATH}")


if __name__ == "__main__":
    build(sys.argv[1:])
