"""Procurement Plans: public STEP exports, one per project version, published through the Documents & Reports API.

Each plan lists every procurement activity (package) with its STEP reference, method, review type, estimated / actual amount,
status, and PLANNED and ACTUAL dates for every stage (terms of reference, notice, bid opening, evaluation, signing, ...). The text
rendition drops empty cells, so a row's dates cannot be mapped to stages from text; the PDF keeps their positions. The parser reads
word positions with PyMuPDF:
  * a section (GOODS, WORKS, NON CONSULTING SERVICES, CONSULTING FIRMS, INDIVIDUAL CONSULTANTS) starts with its title, then a
    block of column labels, then a 'Planned Actual Planned Actual ...' row (which can fall on the next page: labels carry over);
  * the Planned/Actual row gives the x of every date column; the labels above each pair name the stage;
  * labels to the left (Description, Loan, Component, Review Type, Method, ... Process Status) give the field columns;
  * each activity row starts with its STEP reference; every word below it (until the next reference) is assigned by x.
The cells wrap text mid-word ('Employabili / ty'), so descriptions are also kept with spaces removed for matching.

  parse_plan(pdf_path) -> (activities: list[dict], stages: list[dict])
  validate_against_text(activities, text) -> share of activities whose dates equal the text rendition's, in order
"""
from __future__ import annotations
import re
from collections import defaultdict

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REF = re.compile(r"^[A-Z]{2}-[A-Z0-9&.()\-]*\d{4,8}-(?:CW|GO|CS|NC)(?:-[A-Z0-9]+)?")
REF_ANY = re.compile(r"[A-Z]{2}-[A-Z0-9&.()\-]*\d{4,8}-(?:CW|GO|CS|NC)(?:-[A-Z0-9]+)?")
SECTIONS = ("NON CONSULTING SERVICES", "CONSULTING FIRMS", "INDIVIDUAL CONSULTANTS", "GOODS", "WORKS")
STATUSES = ("Under Implementation", "Pending Implementation", "Signed", "Completed", "Canceled", "Cancelled", "Under Review", "Draft",
            "Not Started", "Terminated", "Rejected", "Cleared")
# left-hand field columns, located by the first word of their label
FIELDS = [("description", "Activity"), ("loan", "Loan"), ("component", "Component"), ("review", "Review"), ("method", "Method"),
          ("market", "Market"), ("process", "Procurement"), ("process", "Contract"), ("prequal", "Prequalification"),
          ("estimated", "Estimated"), ("actual", "Actual"), ("status", "Process")]
# canonical stage keys, matched on the label above each Planned/Actual pair (first match wins)
STAGE_KEYS = [
    ("prequal_docs", r"draft pre.?qualification"), ("prequal_eval", r"pre.?qualification eval"),
    ("tor", r"terms of reference"), ("eoi_notice", r"expression of interest"), ("invitation", r"invitation to ident"),
    ("shortlist", r"short ?list"), ("rfp_issued", r"request for proposals as issued"),
    ("tech_opening", r"opening of technical"), ("tech_eval", r"evaluation of technic"), ("combined_eval", r"combined evaluation"),
    ("bid_docs_draft", r"draft bidding doc|justification"), ("notice", r"specific procurement notice|request for quotation|invitation"),
    ("bid_docs_issued", r"bidding documents as issued|as issued"), ("bid_opening", r"submission|opening"),
    ("bid_eval", r"bid evaluation|evaluation report"), ("negotiated", r"negotiated"),
    ("signed", r"signed contract"), ("completion", r"completion"),
]
NOTICE_STAGES = ("eoi_notice", "notice")          # the stages a public procurement notice corresponds to


def stage_key(label: str) -> str:
    low = re.sub(r"\s+", " ", label.lower())
    for key, pat in STAGE_KEYS:
        if re.search(pat, low):
            return key
    return re.sub(r"[^a-z]+", "_", low).strip("_")[:30] or "stage"


def compact(s) -> str:
    """Lower-case letters and digits only: 'C onsultancy Provi sion' and 'Consultancy Provision' compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _num(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _join(ws) -> str:
    """Words of one cell in reading order (PyMuPDF already splits on spaces). Fragments of a word that the cell wrapped onto two
    lines ('Employabili' / 'ty') stay apart: that cannot be undone, which is why matching uses compact()."""
    return " ".join(w[4] for w in sorted(ws, key=lambda w: (round(w[1] / 2.5), w[0])))


def _bands(words, gap=1.0):
    """x-ranges occupied by text across many rows: cells wrap inside their column, so columns are separated by empty gaps."""
    spans = sorted((w[0], w[2]) for w in words)
    out = []
    for a, b in spans:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _section_of(text: str):
    up = re.sub(r"\s+", " ", text.upper()).strip()
    return next((s for s in SECTIONS if up == s or up.startswith(s + " ")), None)


def _build_header(pa_words, label_words):
    """Planned/Actual row + the label block above it -> date columns, stage keys and field columns."""
    pa = sorted(pa_words, key=lambda w: w[0])
    cols = [((w[0] + w[2]) / 2, "planned" if w[4] == "Planned" else "actual") for w in pa]
    pairs, i = [], 0
    while i < len(cols):
        if cols[i][1] == "planned" and i + 1 < len(cols) and cols[i + 1][1] == "actual":
            pairs.append((cols[i][0], cols[i + 1][0])); i += 2
        else:
            pairs.append((cols[i][0], None)); i += 1
    if not pairs:
        return None
    first_x = pairs[0][0]
    stage_words = defaultdict(list)
    for w in label_words:
        xc = (w[0] + w[2]) / 2
        if xc < first_x - 12 or w[4] in ("Planned", "Actual"):
            continue
        k = min(range(len(pairs)), key=lambda j: abs(((pairs[j][0] + (pairs[j][1] or pairs[j][0])) / 2) - xc))
        stage_words[k].append(w)
    labels = [_join(stage_words.get(k, [])) for k in range(len(pairs))]
    columns = []
    for k, (xp, xa) in enumerate(pairs):
        columns.append((xp, "planned", k))
        if xa is not None:
            columns.append((xa, "actual", k))
    left = [w for w in label_words if (w[0] + w[2]) / 2 < first_x - 14]
    fields = []
    for name, first in FIELDS:
        hits = [w for w in left if w[4] == first]
        if hits and name not in [f[0] for f in fields]:
            fields.append((name, min(w[0] for w in hits)))
    fields.sort(key=lambda f: f[1])
    return dict(columns=columns, labels=labels, keys=[stage_key(l) for l in labels], fields=fields, first_x=first_x)


def parse_plan(pdf_path):
    import fitz
    doc = fitz.open(pdf_path)
    acts, stages = [], []
    header = section = None
    carry = None                      # (section, label words) of a section whose Planned/Actual row is on the next page
    for pno, page in enumerate(doc):
        words = page.get_text("words")
        if not words:
            continue
        by_line = defaultdict(list)
        for w in words:
            by_line[round(w[1] / 2.5)].append(w)
        events = []
        for _, ws in sorted(by_line.items()):
            y = min(w[1] for w in ws)
            sec = _section_of(" ".join(w[4] for w in sorted(ws, key=lambda w: w[0])))
            if sec:
                events.append((y, 0, "section", sec))
            if sum(w[4] == "Planned" for w in ws) >= 2:
                events.append((y, 1, "header", [w for w in ws if w[4] in ("Planned", "Actual")]))
        refs = _row_starts(by_line)
        events += [(w[1], 2, "ref", w) for w in refs]
        events.sort(key=lambda e: (e[0], e[1]))
        ref_ys = [w[1] for w in refs]
        page_bands = None
        if refs and header:
            starts = {(round(r[0], 1), round(r[1], 1)) for r in refs}
            body = [w for w in words if w[1] >= refs[0][1] - 2 and (w[0] + w[2]) / 2 < header["first_x"] - 14
                    and (round(w[0], 1), round(w[1], 1)) not in starts and not DATE.match(w[4])]
            page_bands = _bands(body)
        open_section_y = None
        for y, _, kind, val in events:
            if kind == "section":
                section, open_section_y, carry = val, y, None
            elif kind == "header":
                if open_section_y is not None:
                    labels = [w for w in words if open_section_y <= w[1] < y - 0.5]
                elif carry:
                    section, labels = carry
                else:
                    labels = []
                h = _build_header(val, labels)
                if h and not any(h["labels"]) and header and len(header["columns"]) == len(h["columns"]):
                    h = dict(header, columns=h["columns"], first_x=h["first_x"])      # repeated header row: keep the names
                if h:
                    header = h
                open_section_y, carry = None, None
            elif kind == "ref" and header and open_section_y is None:
                i = ref_ys.index(val[1])
                nxt = [e[0] for e in events if e[0] > val[1] + 0.5 and e[2] in ("section", "header", "ref")]
                y_end = min(nxt) if nxt else 10_000
                row = [w for w in words if val[1] - 2 <= w[1] < y_end - 0.5]
                a, s = _row(val, row, header, section, pno + 1, page_bands)
                acts.append(a); stages += s
        if open_section_y is not None:                       # section started, its date row is on the next page
            carry = (section, [w for w in words if w[1] >= open_section_y])
    return acts, stages


def _field_of_band(band, fields):
    """Name a column band by the label nearest its left edge (labels sit a few points right of their data)."""
    if not fields:
        return "description"
    return min(fields, key=lambda f: abs(f[1] - band[0]) + (6 if f[1] < band[0] - 4 else 0))[0]


def _row_starts(by_line):
    """Activity rows start at the left margin with their reference followed by '/': a STEP reference ('KE-SDYA-453045-CS-FBS /
    ...') or an agency's own activity code ('A2 / Improvement of ...', used in some Viet Nam plans). Description lines inside a
    cell also start at the margin but have no ' / ' after their first word.
    Long codes wrap onto several lines ('EOUG/G/2025/202' + '6 / SUPPLY ...', 'GH-MOE-425434-CS-QCB' + 'S / Technical ...'): the
    row is anchored on the line where 'code /' appears, and when that code is only a tail (starts with a digit or '-', or is a
    lone letter) the code pieces on the lines just above are put in front of it."""
    lines = [sorted(ws, key=lambda w: w[0]) for _, ws in sorted(by_line.items())]
    code_like = re.compile(r"[A-Z0-9][A-Z0-9./\-_()&]*")
    starts, used, ref_only = [], set(), []
    for i, ws in enumerate(lines):
        first = ws[0]
        if first[0] > 22:
            continue
        nxt = ws[1][4] if len(ws) > 1 else ""
        if not (nxt == "/" or (first[4].endswith("/") and len(first[4]) > 1)):
            if REF.fullmatch(first[4].rstrip("/")):                 # a complete STEP reference whose '/' sits elsewhere
                ref_only.append(first)
            continue
        parts, start = [first[4].rstrip("/")], first
        tail = parts[0]
        if not re.match(r"[A-Za-z]", tail) or re.fullmatch(r"[A-Za-z]", tail):      # only the end of a wrapped code
            j = i - 1
            while j >= 0 and len(parts) < 4:
                prev = lines[j][0]
                if prev[0] > 22 or start[1] - prev[1] > 7.5 or not code_like.fullmatch(prev[4]) or \
                        (len(lines[j]) > 1 and lines[j][1][4] == "/"):
                    break
                parts.insert(0, prev[4]); used.add(round(prev[1], 1)); start = prev; j -= 1
        code = "".join(parts)
        full_ref = REF.match(code) is not None
        if full_ref or (re.match(r"[A-Za-z]", code) and re.search(r"\d", code) and 2 <= len(code) <= 40):   # 'A2', 'BDCW-13', STEP refs
            starts.append((start[0], start[1], start[2], start[3], code) + tuple(start[5:]))
    taken = {round(s[1], 1) for s in starts} | used
    starts += [(w[0], w[1], w[2], w[3], w[4].rstrip("/")) + tuple(w[5:]) for w in ref_only if round(w[1], 1) not in taken]
    return sorted(starts, key=lambda w: w[1])


def _row(ref_word, row, header, section, page, bands=None):
    ref = ref_word[4]
    left = [w for w in row if (w[0] + w[2]) / 2 < header["first_x"] - 14 and not (abs(w[0] - ref_word[0]) < .5 and abs(w[1] - ref_word[1]) < .5)]
    cells = defaultdict(list)
    fields = header["fields"] or [("description", 0)]
    bands = bands or _bands(left)
    for w in left:
        xc = (w[0] + w[2]) / 2
        band = next((b for b in bands if b[0] - 0.5 <= xc <= b[1] + 0.5), None)
        cells[_field_of_band(band, fields) if band else "description"].append(w)
    for name in [n for n in cells if n not in ("estimated", "actual")]:    # an amount that strayed into a neighbouring band
        amounts = [w for w in cells[name] if re.fullmatch(r"[\d,]+\.\d{2}", w[4])]
        if amounts:
            cells[name] = [w for w in cells[name] if w not in amounts]
            for w in amounts:
                cells["estimated" if "estimated" not in cells else "actual"].append(w)
    txt = {k: _join(v) for k, v in cells.items()}
    desc = re.sub(r"^\s*/\s*", "", txt.get("description", "")).strip()
    m = re.match(r"^(.*?)\s*/\s+(.*)$", desc)                      # a wrapped code's tail ('6 / SUPPLY ...') is not description
    if m and m.group(1) and compact(ref).endswith(compact(m.group(1))):
        desc = m.group(2)
    status_c = compact(txt.get("status"))
    status = next((s for s in STATUSES if compact(s) == status_c or (status_c and compact(s).startswith(status_c[:10]) and len(status_c) >= 6)), None)
    stage_rows, seen = [], set()
    for w in row:
        if not DATE.match(w[4]):
            continue
        xc = (w[0] + w[2]) / 2
        if xc < header["first_x"] - 14:
            continue
        x, kind, k = min(header["columns"], key=lambda c: abs(c[0] - xc))
        if abs(x - xc) > 11 or (k, kind) in seen:            # off the grid, or a second date in one cell: skip
            continue
        seen.add((k, kind))
        stage_rows.append(dict(step_ref=ref, stage_order=k, stage=header["keys"][k], stage_label=header["labels"][k], kind=kind,
                               date=w[4], x=round(xc)))
    stage_rows.sort(key=lambda r: r["x"])
    # amounts have an unmistakable shape: in sections with more columns (works) they sit next to the date grid, so take them
    # from the whole row, left to right: estimated, then actual
    amounts = [_num(w[4]) for w in sorted(row, key=lambda w: w[0]) if re.fullmatch(r"[\d,]+\.\d{2}", w[4])]
    if amounts:
        txt["estimated"] = f"{amounts[0]:.2f}"
        if len(amounts) > 1:
            txt["actual"] = f"{amounts[1]:.2f}"
    m = re.match(r"([A-Z]{2})-(?:[A-Z0-9&.()]+-)*?(\d{4,8})-(?:CW|GO|CS|NC)", ref)
    act = dict(step_ref=ref, step_activity=f"{m.group(1)}-{m.group(2)}" if m else None, section=section, description=desc[:300], description_compact=compact(desc)[:300],
               component=txt.get("component"), review_type=txt.get("review"), method=txt.get("method"), market_approach=txt.get("market"),
               status=status, status_raw=txt.get("status"), estimated_usd=_num(txt.get("estimated", "") or "x"),
               actual_usd=_num(txt.get("actual", "") or "x"), page=page, n_dates=len(stage_rows),
               dates_in_order=[r["date"] for r in stage_rows])
    return act, stage_rows


def fetch_latest(project_id: str, out_dir: str, get_json, get_bytes) -> dict | None:
    """Download the newest Procurement Plan of a project (PDF + text rendition, for validation) into out_dir. Cached."""
    import json, os
    d = get_json("https://search.worldbank.org/api/v3/wds", dict(format="json", rows=1, proid=project_id, docty_exact="Procurement Plan",
                                                                fl="id,docdt,pdfurl,txturl,url", sort="docdt", order="desc"))
    docs = [v for v in (d.get("documents") or {}).values() if isinstance(v, dict) and v.get("id")]
    if not docs or not docs[0].get("pdfurl"):
        return None
    meta = dict(project_id=project_id, plan_doc_id=f"PLAN_{project_id}_{docs[0]['id']}", plan_date=str(docs[0].get("docdt", ""))[:10],
                pdf_url=docs[0]["pdfurl"].replace("http://", "https://"), n_versions=int(d.get("total", 0) or 0))
    os.makedirs(out_dir, exist_ok=True)
    pdf, txt = os.path.join(out_dir, meta["plan_doc_id"] + ".pdf"), os.path.join(out_dir, meta["plan_doc_id"] + ".txt")
    if not os.path.exists(pdf):
        open(pdf, "wb").write(get_bytes(meta["pdf_url"]))
    if not os.path.exists(txt) and docs[0].get("txturl"):
        try:
            open(txt, "wb").write(get_bytes(docs[0]["txturl"]))
        except Exception:
            pass
    json.dump(meta, open(os.path.join(out_dir, meta["plan_doc_id"] + ".json"), "w"))
    return dict(meta, pdf=pdf, txt=txt if os.path.exists(txt) else None)


def validate_against_text(activities, text: str) -> dict:
    """The text rendition keeps each row's dates in reading order (but not their columns): the parse is right when an activity's
    parsed dates, left to right, equal the dates that follow its reference in the text. Returns the share that match exactly."""
    t = re.sub(r"\s+", " ", text)
    refs = sorted({a["step_ref"] for a in activities}, key=len, reverse=True)
    alts = ["(" + r"\s?".join(re.escape(ch) for ch in r) + ")" for r in refs]              # a code may wrap: 'EOUG/G/2025/202 6'
    pat = re.compile(r"(?<![A-Za-z0-9\-])(?:" + "|".join(alts) + r")(?=\s*/)") if refs else None
    pos = [(m.start(), next(refs[k] for k, g in enumerate(m.groups()) if g)) for m in pat.finditer(t)] if pat else []
    by_ref = {}
    for i, (p, ref) in enumerate(pos):
        end = pos[i + 1][0] if i + 1 < len(pos) else len(t)
        by_ref.setdefault(ref, re.findall(r"\d{4}-\d{2}-\d{2}", t[p:end]))
    ok = n = 0
    bad, checked = [], set()
    for a in activities:
        if a["step_ref"] not in by_ref:
            continue
        n += 1; checked.add(a["step_ref"])
        if by_ref[a["step_ref"]] == a["dates_in_order"]:
            ok += 1
        else:
            bad.append(a["step_ref"])
    return dict(checked=n, exact=ok, share=round(ok / n, 3) if n else None, mismatches=bad[:5], bad_refs=set(bad), checked_refs=checked)
