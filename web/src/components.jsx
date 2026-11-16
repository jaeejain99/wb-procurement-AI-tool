import { useState } from "react";
import { api } from "./api";
import { dur } from "./plain";

export const go = (route) => { window.location.hash = route; };
export const openPassage = (id) => go(`/library?passage=${encodeURIComponent(id)}`);

export function Spinner({ label }) {
  return <span className="row small muted"><span className="spinner" />{label}</span>;
}

export function Skeleton({ lines = 3 }) {
  return <div>{Array.from({ length: lines }, (_, i) => <div key={i} className="skeleton" style={{ width: `${90 - i * 12}%` }} />)}</div>;
}

export function ErrorBox({ error, onRetry }) {
  if (!error) return null;
  return (
    <div className="card" role="alert" style={{ borderColor: "var(--critical)" }}>
      <div className="row"><span className="tag critical">✕ Error</span><span className="small">{error}</span>
        {onRetry && <button className="btn ghost" onClick={onRetry}>Try again</button>}</div>
    </div>
  );
}

export function Empty({ title, children }) {
  return <div className="empty"><div style={{ fontWeight: 600, color: "var(--ink)" }}>{title}</div><div className="small">{children}</div></div>;
}

// Status always carries an icon + a word, never colour alone.
const STATUS = { good: "✓", warning: "!", serious: "▲", critical: "✕" };
export function Status({ level, children }) {
  return <span className={`tag ${level}`}><span aria-hidden>{STATUS[level]}</span>{children}</span>;
}
export function riskLevel(score) {
  if (score >= 0.6) return ["critical", "High risk"];
  if (score >= 0.3) return ["serious", "Watch"];
  if (score > 0) return ["warning", "Minor"];
  return ["good", "No warning signs"];
}

export function Stat({ label, value, sub }) {
  return <div className="stat"><div className="label">{label}</div><div className="value">{value ?? "–"}</div>{sub && <div className="sub">{sub}</div>}</div>;
}

// Planning range on one track: fast case (P10) - usual (median) - slow case (P90), labelled in plain words.
export function RangeBar({ p10, p50, p90 }) {
  const hi = Math.max(p90 * 1.12, 1), pc = (x) => `${Math.min(100, (x / hi) * 100)}%`;
  return (
    <div className="range" role="img" aria-label={`fast case ${dur(p10)}, usually ${dur(p50)}, up to ${dur(p90)}`}>
      <div className="track" />
      <div className="band" style={{ left: pc(p10), width: `calc(${pc(p90)} - ${pc(p10)})` }} title={`8 in 10 similar tenders: ${p10}–${p90} days`} />
      <div className="mid" style={{ left: pc(p50) }} title={`Median ${p50} days`} />
      <span className="lab" style={{ left: pc(p10) }}>fast: {dur(p10)}</span>
      <span className="lab top" style={{ left: pc(p50) }}>usually {dur(p50)}</span>
      <span className="lab" style={{ left: pc(p90) }}>up to {dur(p90)}</span>
    </div>
  );
}

// ISR development-objective rating over time: rating names on the side (6 Highly Satisfactory .. 1 Highly Unsatisfactory),
// one dot per status report placed by its date, the unsatisfactory range (3 and below) shaded and labelled.
const RATINGS = ["Highly Satisfactory", "Satisfactory", "Moderately Satisfactory", "Moderately Unsatisfactory", "Unsatisfactory", "Highly Unsatisfactory"];
const monthYear = (d) => new Date(d).toLocaleDateString("en-GB", { month: "short", year: "numeric" });
const longDate = (d) => new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });

export function IsrTimeline({ points }) {
  if (!points?.length) return <p className="small muted">No progress ratings found in this project's status reports.</p>;
  const pts = points.slice(-16);
  const t = pts.map((p) => new Date(p.date).getTime());
  const span = t[t.length - 1] - t[0];
  const x = (i) => (span ? 4 + ((t[i] - t[0]) / span) * 92 : 50);            // % across, by date
  const y = (score) => ((6 - score) / 5) * 100;                               // % down: 6 at the top, 1 at the bottom
  const ticks = pts.length <= 4 ? pts.map((_, i) => i) : [0, Math.round((pts.length - 1) / 3), Math.round((2 * (pts.length - 1)) / 3), pts.length - 1];
  const last = pts[pts.length - 1];
  return (
    <div>
      <div className="isr-chart" role="img" aria-label={`Progress ratings in ${pts.length} status reports, latest ${last.dev_objective}`}>
        <div className="isr-y">{RATINGS.map((r, i) => <span key={r} style={{ top: `${(i / 5) * 100}%` }}>{r}</span>)}</div>
        <div className="isr-plot">
          {RATINGS.map((r, i) => <div key={r} className="isr-grid" style={{ top: `${(i / 5) * 100}%` }} />)}
          <div className="isr-band" style={{ top: `${y(3.5)}%` }}><span>Below Moderately Satisfactory</span></div>
          <svg className="isr-line" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
            <polyline points={pts.map((p, i) => `${x(i)},${y(p.score)}`).join(" ")} vectorEffect="non-scaling-stroke" />
          </svg>
          {pts.map((p, i) => (
            <div key={i} className={`isr-dot${p.score <= 3 ? " low" : ""}`} style={{ left: `${x(i)}%`, top: `${y(p.score)}%` }} tabIndex={0}
              title={`${longDate(p.date)}: ${p.dev_objective}`} aria-label={`${longDate(p.date)}: ${p.dev_objective}`} />
          ))}
        </div>
        <div />
        <div className="isr-x">{ticks.map((i) => <span key={i} style={{ left: `${x(i)}%` }}>{monthYear(pts[i].date)}</span>)}</div>
      </div>
      <p className="tiny" style={{ marginTop: 6 }}>Latest: <b>{last.dev_objective}</b> ({longDate(last.date)}) · {pts.length} status report{pts.length === 1 ? "" : "s"}{points.length > pts.length ? ` (latest ${pts.length} of ${points.length})` : ""}. Hover a dot for its date and rating.</p>
      <details className="tiny"><summary>Show as table</summary>
        <table className="tbl"><thead><tr><th>Status report date</th><th>Progress rating</th></tr></thead>
          <tbody>{[...pts].reverse().map((p, i) => <tr key={i}><td>{longDate(p.date)}</td><td>{p.dev_objective}</td></tr>)}</tbody></table></details>
    </div>
  );
}

function Feedback({ ids, question, surface }) {
  const [state, setState] = useState(null);
  if (!ids?.length) return null;
  const send = async (useful) => {
    try { await api.feedback({ chunk_ids: ids, useful, surface, question: question?.slice(0, 300) }); setState(useful ? "up" : "down"); }
    catch (e) { setState(e.message); }
  };
  if (state) return <span className="tiny">{state === "up" || state === "down" ? "Thanks — this tunes which passages come first." : state}</span>;
  return <span className="row tiny">Useful? <button className="btn ghost" onClick={() => send(true)} aria-label="Useful">👍</button>
    <button className="btn ghost" onClick={() => send(false)} aria-label="Not useful">👎</button></span>;
}
export { Feedback };

// A verified agent answer: sentences with numbered citations and their sources (removed sentences and the trace stay server-side
// in the event log, not on screen).
export function Answer({ r, surface = "agent" }) {
  if (r.refused) {
    return <div className="card"><Status level="warning">Declined</Status> <span>{r.answer?.[0]?.text}</span>
      <p className="tiny">{r.refusal_reason === "model" ? "The agent declined this request." : "A policy check declined this before any model was called."}</p></div>;
  }
  if (r.mode === "fallback") {
    return <div className="card"><Status level="warning">Model unavailable</Status>
      <p className="small">Showing the most relevant passages instead; no text was generated.</p>
      {(r.passages || []).map((c) => <blockquote key={c.id} className="quote">{c.text}<div className="tiny">{c.doc_type} {c.project_id} · page {c.page}</div></blockquote>)}</div>;
  }
  const ids = [...new Set(r.answer.flatMap((a) => a.ids))];
  const n = (id) => ids.indexOf(id) + 1;
  const facts = r.answer.filter((a) => !a.meta);
  return (
    <div className="card msg-a">
      {r.held_back ? <p className="muted">No sentence survived verification, so the draft is held back rather than shown unchecked.</p>
        : r.answer.map((a, i) => a.meta && !a.ids.length
          ? <p key={i} className="sent" style={{ fontWeight: 600 }}>{a.text}</p>
          : <p key={i} className="sent">{a.text}{a.ids.map((id) => <button key={id} className="cite" title={r.sources[id]?.label || id}
              onClick={() => r.sources[id]?.kind === "passage" ? openPassage(id) : document.getElementById(`src-${r.turnKey}-${n(id)}`)?.scrollIntoView({ behavior: "smooth" })}>{n(id)}</button>)}</p>)}
      <div className="spread" style={{ marginTop: 8 }}>
        <span className="tiny">✓ {facts.length} sentence{facts.length === 1 ? "" : "s"} verified · {r.seconds}s</span>
        <Feedback ids={ids} question={r.question} surface={surface} />
      </div>
      {ids.length > 0 && <details><summary>Sources ({ids.length})</summary>
        {ids.map((id) => { const s = r.sources[id] || {}; return (
          <div key={id} id={`src-${r.turnKey}-${n(id)}`} className="removed">
            <div className="row"><span className="cite">{n(id)}</span><b className="small">{s.label || id}</b>{s.country && <span className="tiny">{s.country}</span>}
              {s.kind === "passage" && <button className="btn ghost small" onClick={() => openPassage(id)}>Open in library →</button>}</div>
            {s.kind === "passage" ? <blockquote className="quote">{s.text}</blockquote>
              : <div className="small" style={{ whiteSpace: "pre-line", marginTop: 4 }}>{s.display || s.text}</div>}
          </div>); })}
      </details>}
    </div>
  );
}
