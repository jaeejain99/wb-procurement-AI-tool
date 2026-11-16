import { useEffect, useState } from "react";
import { api } from "../api";
import { Empty, ErrorBox, Skeleton, Spinner, go } from "../components";

const SECTIONS = [["", "Any section"], ["lessons", "Lessons learned"], ["procurement", "Procurement & fiduciary"], ["ratings", "Ratings"], ["context", "Context & objectives"]];
const DOC = { ICR: "Completion report", PAD: "Appraisal document", ISR: "Status report", PPAR: "IEG evaluation" };

// Same filler words and stemmer as the server's relevance threshold (services.key_terms), so what is highlighted is exactly
// what made a passage count as a match: "delays" also marks "delayed", "bids" marks "bid".
const STOP = new Set(("the and for are was were with that this from into than then they them their there what which when where while who whom " +
  "whose why how about after before over under between because during does did done have has had been being also only very " +
  "more most such some any each other these those your our its not but can could would should will may might must shall far " +
  "led lead all one two get got say says said show shows").split(" "));

function stem(w) {
  if (w.length > 5) { const s = w.replace(/(ies|es|s|ing|ed|ions|ion)$/, ""); if (s.length >= 4) w = s; }
  else if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1);
  return w.length > 4 && w.endsWith("e") ? w.slice(0, -1) : w;
}
const ok = (w) => (w.length >= 3 || /^\d+$/.test(w)) && !STOP.has(w);
const HY = /[-‐‑]/;
// Same rules as the server (services.token_terms): in a query, "re-tendering" / "e-procurement" are ONE term; "cost-benefit" is two.
function tokenTerms(tok, query) {
  const parts = tok.split(HY);
  if (parts.length === 1) return ok(tok) ? [stem(tok)] : [];
  const joined = stem(parts.join("")), words = parts.filter(ok).map(stem);
  if (query) return parts.some((p) => p.length < 3 && !/^\d+$/.test(p)) ? [joined] : words;
  return [...words, joined];
}
const TOKEN = /([A-Za-z0-9]+(?:[-‐‑][A-Za-z0-9]+)*)/;
const terms = (s) => new Set(((s || "").toLowerCase().match(new RegExp(TOKEN.source, "g")) || []).flatMap((t) => tokenTerms(t, true)));

export function highlight(text, q) {
  const qt = terms(q);
  if (!qt.size || !text) return text;
  return text.split(TOKEN).map((part, i) => {
    if (!(i % 2)) return part;
    const t = part.toLowerCase(), parts = part.split(HY);
    if (parts.length > 1 && qt.has(stem(parts.join("").toLowerCase()))) return <mark key={i}>{part}</mark>;   // the whole compound matched
    if (parts.length === 1) return ok(t) && qt.has(stem(t)) ? <mark key={i}>{part}</mark> : part;
    let pos = 0;                                                                                              // mark matching parts only
    return <span key={i}>{parts.map((p, j) => {
      const sep = j ? part.slice(pos - 1, pos) : ""; const lw = p.toLowerCase(); pos += p.length + 1;
      return <span key={j}>{sep}{ok(lw) && qt.has(stem(lw)) ? <mark>{p}</mark> : p}</span>;
    })}</span>;
  });
}

export const PdfLink = ({ url, page, children }) => url
  ? <a className="btn ghost small" href={url} target="_blank" rel="noopener noreferrer" title="Opens the official World Bank PDF at this page">{children || `Original PDF, page ${page}`} ↗</a>
  : null;

// The whole report, page by page, opened at one passage: that passage is highlighted and scrolled into view, search words marked.
function DocumentView({ id, q }) {
  const docId = id.split("#")[0];
  const [d, setD] = useState(null), [err, setErr] = useState(null);
  useEffect(() => { setD(null); setErr(null); api.document(docId).then(setD).catch((e) => setErr(e.message)); }, [docId]);
  useEffect(() => {
    if (d) requestAnimationFrame(() => document.getElementById(`psg-${id}`)?.scrollIntoView({ block: "center" }));
  }, [d, id]);
  if (err) return <ErrorBox error={err} />;
  if (!d) return <div className="card"><Skeleton lines={8} /></div>;
  const target = d.pages.flatMap((pg) => pg.passages.map((p) => ({ ...p, page: pg.page, pdf: pg.pdf_page_url }))).find((p) => p.chunk_id === id);
  let lastSection = null;
  return (
    <>
      <div className="card doc-head">
        <div className="spread"><div><div className="tiny">{DOC[d.doc_type] || d.doc_type} · project {d.project_id} · {d.country} · {d.date}</div>
          <h2 style={{ margin: "2px 0" }}>{target?.section || d.doc_id}</h2>
          <div className="small muted">{target ? `Page ${target.page} · ` : ""}{d.doc_id} · {d.pages.length} pages with text</div></div>
          <div className="row"><PdfLink url={target?.pdf || d.pdf_url} page={target?.page}>{target ? undefined : "Original PDF"}</PdfLink>
            <button className="btn" onClick={() => go("/library")}>← Back to search</button></div></div>
        {target && <div className="spread" style={{ marginTop: 8 }}><span className="tiny">The highlighted passage is the one you opened; search words are marked. Text as extracted from the report.</span></div>}
      </div>
      <div className="card doc-body">
        {d.pages.map((pg) => (
          <section key={pg.page} className="doc-page">
            <div className="doc-page-no"><span>Page {pg.page}</span><PdfLink url={pg.pdf_page_url} page={pg.page}>PDF</PdfLink></div>
            {pg.passages.map((p) => {
              const head = p.section !== lastSection ? (lastSection = p.section) : null;
              return (
                <div key={p.chunk_id}>
                  {head && <div className="doc-section">{head}</div>}
                  <div id={`psg-${p.chunk_id}`} className={`doc-psg${p.kind === "table" ? " table" : ""}${p.chunk_id === id ? " target" : ""}`}>{highlight(p.text, q)}</div>
                </div>);
            })}
          </section>))}
      </div>
    </>
  );
}

export default function EvidenceLibrary({ meta, params }) {
  const saved = (() => { try { return JSON.parse(sessionStorage.getItem("procintel.library")) || {}; } catch { return {}; } })();
  const [q, setQ] = useState(params.q || saved.q || ""), [docType, setDocType] = useState(saved.docType || ""), [section, setSection] = useState(saved.section || ""), [country, setCountry] = useState(saved.country || "");
  const [res, setRes] = useState(saved.res || null), [busy, setBusy] = useState(false), [err, setErr] = useState(null);
  useEffect(() => { sessionStorage.setItem("procintel.library", JSON.stringify({ q, docType, section, country, res })); }, [q, docType, section, country, res]);
  const search = async (query, f) => {
    if (query.trim().length < 3) return;
    setBusy(true); setErr(null);
    try { setRes({ ...(await api.search({ q: query.trim(), k: 12, doc_type: f.docType, section: f.section, country: f.country })), q: query.trim(), ...f }); }
    catch (x) { setErr(x.message); }
    setBusy(false);
  };
  const run = (e) => { e?.preventDefault(); search(q, { docType, section, country }); };
  // Changing a filter re-runs the last search straight away (with the query that was searched, not a half-typed one).
  const setFilter = (set, key) => (e) => {
    const v = e.target.value; set(v);
    if (res?.q) search(res.q, { docType, section, country, [key]: v });
  };
  const shownQ = res?.q || "";
  if (params.passage) return <><div className="page-head"><h1>Evidence library</h1><p>The full report, opened at the passage.</p></div><DocumentView id={params.passage} q={params.q || shownQ} /></>;
  return (
    <>
      <div className="page-head"><h1>Evidence library</h1><p>Search project reports directly. Results are quotes, nothing is generated; open any passage to read it in context.</p></div>
      <form className="card" onSubmit={run}>
        <div className="row"><input style={{ flex: 3, minWidth: 260 }} value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. bid prices far above the cost estimate led to re-tendering" aria-label="Search" />
          <button className="btn primary" disabled={busy || q.trim().length < 3}>{busy ? <Spinner /> : "Search"}</button></div>
        <div className="row" style={{ marginTop: 10 }}>
          <select style={{ width: "auto" }} value={docType} onChange={setFilter(setDocType, "docType")} aria-label="Document type"><option value="">All documents</option>
            {meta.options.doc_types.map((d) => <option key={d} value={d}>{DOC[d] || d}</option>)}</select>
          <select style={{ width: "auto" }} value={section} onChange={setFilter(setSection, "section")} aria-label="Section">{SECTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
          <select style={{ width: "auto" }} value={country} onChange={setFilter(setCountry, "country")} aria-label="Country"><option value="">All countries</option>
            {meta.options.countries.map((c) => <option key={c.code} value={c.name}>{c.name}</option>)}</select>
        </div>
      </form>
      <ErrorBox error={err} />
      {res?.results.length > 0 && <p className="tiny" style={{ margin: "10px 2px" }}>{res.results.length} passages for “{res.q}”</p>}
      {res && !res.results.length && <div className="card"><Empty title="No results found">
        No passage contains enough of your search terms{res.country || res.docType || res.section ? " with these filters" : ""}. Try other words{res.country || res.docType || res.section ? " or remove a filter" : ""}.</Empty></div>}
      <div className="stack">{res?.results.map((c) => (
        <div key={c.chunk_id} className="card">
          <div className="spread"><div className="row"><span className="tag accent">{DOC[c.doc_type] || c.doc_type}</span><span className="small"><b>{c.project_id}</b> · {c.country} · page {c.page}</span>{c.match != null && <span className="tag" title="Share of the search-term weight found in this passage (rare words count more)">match {Math.round(c.match * 100)}%</span>}</div>
            <div className="row"><button className="btn ghost small" onClick={() => go(`/library?passage=${encodeURIComponent(c.chunk_id)}&q=${encodeURIComponent(shownQ)}`)}>Read in context →</button><PdfLink url={c.pdf_page_url} page={c.page} /></div></div>
          <div className="tiny" style={{ marginTop: 4 }}>{c.section}</div>
          <blockquote className="quote">{highlight(c.text, shownQ)}</blockquote>
        </div>))}</div>
    </>
  );
}
