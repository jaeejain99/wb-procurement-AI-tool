import { useEffect, useState } from "react";
import { api } from "../api";
import { Empty, ErrorBox, IsrTimeline, Skeleton, Stat, Status, go, openPassage, riskLevel } from "../components";
import { RISK_HELP, dur } from "../plain";

const LEVEL = { critical: "critical", serious: "serious", warning: "warning" };
// One warning sign in plain words; the exact technical statement is kept on hover for specialists.
function Alert({ a }) {
  return <div className="alert" title={a.tech}><div className="row"><Status level={LEVEL[a.level] || "warning"}>{a.title}</Status></div>
    <div className="small" style={{ marginTop: 4 }}>{a.detail}</div>{a.note && <div className="tiny muted" style={{ marginTop: 2 }}>{a.note}</div>}
    {a.packages?.length > 1 && <details className="tiny" style={{ marginTop: 4 }}><summary>All {a.packages.length} packages behind plan</summary>
      <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>{a.packages.map((x) => <li key={x.step_ref}>
        <span title="Procurement Plan activity reference">{x.step_ref}</span> · {x.description || "–"} · signing planned {x.signed_planned} · {dur(x.days_late)} behind
        {x.status ? ` · ${x.status}` : ""}</li>)}</ul></details>}</div>;
}
const conflictText = (c) => `The last status report (${c.isr_last.date}) rated progress ${c.isr_last.rating}; the completion report rated the outcome ${c.icr.rating}. Tenderwise uses the completion report's rating, as the final assessment. ${c.note.split(";")[0]}.`;

function ProjectPanel({ pid, onClose }) {
  const [p, setP] = useState(null), [err, setErr] = useState(null);
  useEffect(() => { setP(null); setErr(null); api.project(pid).then(setP).catch((e) => setErr(e.message)); }, [pid]);
  useEffect(() => { const k = (e) => e.key === "Escape" && onClose(); window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k); }, [onClose]);
  const ask = () => go(`/ask?q=${encodeURIComponent(`Why is project ${pid} flagged, and what should the team do about it?`)}`);
  return (
    <>
      <div className="drawer-bg" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`Project ${pid}`}>
        <div className="spread"><span className="tiny">Project</span><button className="btn ghost" onClick={onClose} aria-label="Close">✕ Close</button></div>
        <ErrorBox error={err} />
        {!p && !err && <Skeleton lines={8} />}
        {p && <div className="stack">
          <div><h2 style={{ margin: "2px 0" }}>{p.name}</h2>
            <div className="row small muted">{p.project_id} · {p.country} · {p.sector} · <span className="tag">{p.status}</span></div></div>
          <div className="stats">
            <Stat label="Commitment" value={p.commitment_usd_m != null ? `$${p.commitment_usd_m}m` : "–"} />
            <Stat label="Approved" value={p.approval_date || "–"} sub={`closes ${p.closing_date || "–"}`} />
            <Stat label="Contracts" value={p.contracts.n} />
          </div>
          {p.warnings && <div className="card"><div className="spread"><h2 style={{ margin: 0 }}>Warning signs</h2>
            {(() => { const [l, t] = riskLevel(p.warnings.risk_score); return <span title={RISK_HELP[t]}><Status level={l}>{t}</Status></span>; })()}</div>
            {p.warnings.alerts?.length ? <div className="stack" style={{ gap: 10, marginTop: 8 }}>{p.warnings.alerts.map((a, i) => <Alert key={i} a={a} />)}</div> : <p className="small muted">No warning signs.</p>}
            <button className="btn" onClick={ask}>💬 Ask why</button></div>}
          <div className="card"><h2>Progress ratings in status reports</h2><IsrTimeline points={p.isr} />
            {p.icr_outcome && <p className="small">Final outcome rating (completion report): <b>{p.icr_outcome}</b></p>}
            {p.conflict && <div className="removed" title={`rule ${p.conflict.rule_applied}; ${p.conflict.note}`}><Status level={p.conflict.severity === "major" ? "serious" : "warning"}>{p.conflict.severity === "major" ? "Ratings disagree strongly" : "Ratings disagree slightly"}</Status>
              <p className="small">{conflictText(p.conflict)}</p></div>}
          </div>
          <div className="card"><h2>Tenders still open ({p.open_tenders.length})</h2>
            {p.open_tenders.length ? <table className="tbl"><thead><tr><th>Tender</th><th className="num">Open for</th><th className="num">Similar tenders are signed in</th><th></th></tr></thead>
              <tbody>{p.open_tenders.map((t) => <tr key={t.tender_id}><td className="small">{t.description || t.tender_id}<div className="tiny">{t.step_ref ? <><span title="STEP procurement reference">{t.step_ref}</span> · </> : null}{t.category} · {t.method} · noticed {t.notice_date}</div></td>
                <td className="num" title={`${t.days_open} days`}>{dur(t.days_open)}</td><td className="num" title={`median ${t.p50_days} days; 90% within ${t.p90_days} days`}>usually {dur(t.p50_days)}<div className="tiny">9 in 10 within {dur(t.p90_days)}</div></td>
                <td>{t.overdue ? <Status level="critical">Running late</Status> : t.advertising ? <span title={`Bids or expressions of interest due ${t.submission_deadline}`}><Status level="good">Being advertised</Status></span> : <Status level="good">On track</Status>}</td></tr>)}</tbody></table>
              : <p className="small muted">No open tenders recorded.</p>}</div>
          {p.lessons.length > 0 && <div className="card"><h2>Lessons and procurement notes from this project</h2>
            {p.lessons.map((l) => <div key={l.chunk_id} className="removed"><div className="spread"><span className="tiny">{l.doc_type} · page {l.page} · {l.section}</span>
              <button className="btn ghost small" onClick={() => openPassage(l.chunk_id)}>Open →</button></div><blockquote className="quote">{l.text}</blockquote></div>)}</div>}
          <details className="card"><summary>Documents ({p.documents.length})</summary>
            <ul className="small">{p.documents.map((d) => <li key={d.doc_id}>{d.doc_type} · {d.date || "undated"} · <span className="tiny">{d.doc_id}</span></li>)}</ul></details>
        </div>}
      </aside>
    </>
  );
}

export default function PortfolioMonitor({ meta, params }) {
  const [country, setCountry] = useState(params.country || "");
  const [data, setData] = useState(null), [err, setErr] = useState(null), [open, setOpen] = useState(params.project || null);
  const [all, setAll] = useState(false);
  const load = () => { setData(null); setErr(null); api.portfolio({ country, top: 200 }).then(setData).catch((e) => setErr(e.message)); };
  useEffect(load, [country]);
  return (
    <>
      <div className="page-head spread"><div><h1>Portfolio monitor</h1>
        <p>Active projects ranked by risk. Every warning says what triggered it; open a project for the details.</p></div>
        <label className="field" style={{ minWidth: 220, marginTop: 0 }}><span>Country</span>
          <select value={country} onChange={(e) => setCountry(e.target.value)}><option value="">All countries</option>
            {meta.options.countries.map((c) => <option key={c.code} value={c.code}>{c.name}</option>)}</select></label></div>
      <ErrorBox error={err} onRetry={load} />
      {!data && !err && <div className="card"><Skeleton lines={8} /></div>}
      {data && <>
        <div className="stats centered" style={{ marginBottom: 14 }}>
          <Stat label="Active Projects" value={data.total_active} />
          <Stat label="With Warning Signs" value={data.flagged} sub="projects with at least one warning sign" />
          <Stat label="Tenders Running Late" value={data.overdue_tenders} />
        </div>
        <div className="spread" style={{ margin: "0 2px 8px" }}>
          <span className="small muted">{all ? `All ${data.projects.length} active projects` : `${data.projects.filter((p) => p.flags.length).length} projects with warning signs`}</span>
          <div className="seg"><button className={!all ? "on" : ""} onClick={() => setAll(false)}>With warning signs</button><button className={all ? "on" : ""} onClick={() => setAll(true)}>Show all</button></div></div>
        <div className="card" style={{ padding: 0, overflowX: "auto" }}>
          {!data.projects.filter((p) => all || p.flags.length).length ? <Empty title={all ? "No active projects" : "No warning signs"}>{all ? "Try another country." : "Nothing is flagged here. Use “Show all” to browse every active project."}</Empty> :
            <table className="tbl"><thead><tr><th>Project</th><th>Risk</th><th className="num">Warning Count</th><th className="num">Open tenders</th><th>Latest progress rating</th></tr></thead>
              <tbody>{data.projects.filter((p) => all || p.flags.length).map((p) => { const [l, t] = riskLevel(p.risk_score); return (
                <tr key={p.project_id} className={`click ${open === p.project_id ? "sel" : ""}`} onClick={() => setOpen(p.project_id)} tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && setOpen(p.project_id)}>
                  <td><b>{p.name}</b><div className="tiny">{p.project_id} · {p.country} · {p.sector}</div></td>
                  <td title={RISK_HELP[t]}><Status level={l}>{t}</Status></td>
                  <td className="num" title={p.alerts.length ? "Open the project to see each warning" : "No warning signs"}>{p.alerts.length}</td>
                  <td className="num">{p.open_tenders}{p.overdue_tenders > 0 && <div><Status level="critical">{p.overdue_tenders} running late</Status></div>}</td>
                  <td className="small">{p.last_isr || "–"}</td></tr>); })}</tbody></table>}
        </div>
      </>}
      {open && <ProjectPanel pid={open} onClose={() => setOpen(null)} />}
    </>
  );
}
