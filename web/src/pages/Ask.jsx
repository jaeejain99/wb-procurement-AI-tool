import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Answer, ErrorBox, Spinner, go } from "../components";

const SUGGEST = [
  "We are about to tender USD 20m of rural road rehabilitation works in Kenya. What should we plan for?",
  "What do completion reports say about land acquisition delays in works contracts?",
  "How long do consultant selections take in Ghana, and what slows them down?",
  "Which active projects in Bangladesh show early-warning signs, and why?",
];
const KEY = "procintel.ask";

const TOOL = { search_documents: "Searched reports", forecast_package: "Forecast the schedule", get_project: "Read the project record",
  early_warnings: "Checked early warnings", list_values: "Looked up valid values" };

// What the user sees while the answer is built: steps, tools, and ONLY sentences that already passed verification.
function LiveAnswer({ live }) {
  const ids = [...new Set(live.sentences.flatMap((s) => s.ids))];
  return (
    <div className="card msg-a">
      <div className="row small muted"><span className="spinner" />{live.status}</div>
      {live.tools.length > 0 && <div className="row" style={{ margin: "8px 0" }}>{live.tools.map((t, i) =>
        <span key={i} className="tag" title={JSON.stringify(t.args)}>✓ {TOOL[t.name] || t.name} · {t.summary} · {t.seconds}s</span>)}</div>}
      {live.sentences.map((s, i) => s.meta && !s.ids.length
        ? <p key={i} className="sent" style={{ fontWeight: 600 }}>{s.text}</p>
        : <p key={i} className="sent">{s.text}{s.ids.map((id) => <span key={id} className="cite" title={id}>{ids.indexOf(id) + 1}</span>)}</p>)}
      <div className="tiny">✓ {live.sentences.filter((s) => !s.meta).length} verified so far</div>
    </div>
  );
}

function loadConv() {               // a turn interrupted by a reload has no final answer: drop it rather than spin forever
  try { const c = JSON.parse(sessionStorage.getItem(KEY)) || { id: null, turns: [], context: null }; return { ...c, turns: c.turns.filter((t) => t.r) }; }
  catch { return { id: null, turns: [], context: null }; }
}

export default function Ask({ params, pkg }) {
  const [conv, setConv] = useState(loadConv);
  const [text, setText] = useState(params.q || "");
  const [busy, setBusy] = useState(false), [err, setErr] = useState(null);
  const bottom = useRef(null), sentPrefill = useRef(false);

  useEffect(() => { sessionStorage.setItem(KEY, JSON.stringify(conv)); }, [conv]);
  useEffect(() => {                                    // opened from "Ask about this package": start a fresh conversation with that context
    if (params.from === "package" && pkg && conv.context?.description !== pkg.description) {
      const { report, ...ctx } = pkg; setConv({ id: null, turns: [], context: ctx });
    }
  }, [params.from, pkg]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [conv.turns.length, busy]);

  const send = async (q) => {
    q = (q ?? text).trim(); if (q.length < 5 || busy) return;
    setBusy(true); setErr(null); setText("");
    const fresh = { status: "Planning which tools this question needs…", tools: [], sentences: [], removed: 0 };
    setConv((c) => ({ ...c, turns: [...c.turns, { q, r: null, live: fresh }] }));
    const patchLive = (f) => setConv((c) => ({ ...c, turns: c.turns.map((t, i) => (i === c.turns.length - 1 ? { ...t, live: f(t.live) } : t)) }));
    const body = { question: q, conversation_id: conv.id, context: conv.turns.length ? null : conv.context };
    let gotEvent = false;
    try {
      let r;
      try {
        r = await api.askStream(body, (ev) => {
          gotEvent = true;
          if (ev.type === "status") patchLive((l) => ({ ...l, status: ev.text }));
          else if (ev.type === "tool") patchLive((l) => ({ ...l, tools: [...l.tools, ev] }));
          else if (ev.type === "sentence") patchLive((l) => ({ ...l, status: "Verifying each sentence before showing it…", sentences: [...l.sentences, ev] }));
          else if (ev.type === "removed") patchLive((l) => ({ ...l, removed: l.removed + 1 }));
        });
      } catch (e) {
        if (gotEvent || e.status === 422 || e.status === 503) throw e;  // a real error, not a streaming problem
        r = await api.ask(body);                                             // e.g. a proxy that buffers streams
      }
      setConv((c) => ({ ...c, id: r.conversation_id, turns: c.turns.map((t, i) => (i === c.turns.length - 1 ? { q, r } : t)) }));
    } catch (e) {
      setErr(e.message); setConv((c) => ({ ...c, turns: c.turns.slice(0, -1) })); setText(q);
    }
    setBusy(false);
  };
  useEffect(() => { if (params.q && !sentPrefill.current) { sentPrefill.current = true; send(params.q); go("/ask"); } }, [params.q]);

  const reset = () => { setConv({ id: null, turns: [], context: null }); setErr(null); };
  const onKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };

  return (
    <>
      <div className="page-head spread"><div><h1>Ask</h1>
        <p>Ask in plain words and follow up. Every sentence is checked against its sources before you see it.</p></div>
        <button className="btn" onClick={reset} disabled={busy}>＋ New conversation</button></div>

      {conv.context && <div className="card" style={{ marginBottom: 12 }}><div className="spread">
        <span className="small">Talking about your package: <b>{conv.context.category} · {conv.context.sector} · {conv.context.country}{conv.context.value_usd_m ? ` · USD ${conv.context.value_usd_m}m` : ""}</b></span>
        <button className="btn ghost small" onClick={() => setConv((c) => ({ ...c, context: null }))}>Remove context</button></div></div>}

      <div className="chat">
        {!conv.turns.length && <div className="card">
          <p className="muted" style={{ marginTop: 0 }}>Try one of these, or ask your own. You can follow up with questions like “What about Ghana?” or “How did they fix it?”.</p>
          <div className="suggest">{SUGGEST.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}</div></div>}
        {conv.turns.map((t, i) => (
          <div key={i} className="chat">
            <div className="msg-q">{t.q}</div>
            {t.r ? <Answer r={{ ...t.r, turnKey: i }} /> : t.live ? <LiveAnswer live={t.live} /> : <div className="card"><Spinner label="Working…" /></div>}
          </div>
        ))}
        <div ref={bottom} />
      </div>

      <ErrorBox error={err} />
      <div className="composer">
        <div className="box">
          <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={onKey} maxLength={1000} aria-label="Your question"
            placeholder={conv.turns.length ? "Ask a follow-up…" : "Ask about packages, projects, lessons or timelines…"} />
          <button className="btn primary" onClick={() => send()} disabled={busy || text.trim().length < 5}>{busy ? <span className="spinner" /> : "Send"}</button>
        </div>
        <div className="tiny" style={{ marginTop: 4 }}>{conv.turns.length ? `${Math.min(conv.turns.length, 4)} earlier answer(s) are used for follow-ups.` : "Enter to send · Shift+Enter for a new line"}
          {" "}Tenderwise does not rank or blacklist suppliers, predict tender winners, or give legal advice.</div>
      </div>
    </>
  );
}
