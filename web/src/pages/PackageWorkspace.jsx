import { useMemo, useState } from "react";
import { api } from "../api";
import { Empty, ErrorBox, Feedback, RangeBar, Skeleton, Spinner, Stat, Status, go, openPassage } from "../components";
import { dur } from "../plain";

const THEME = { estimate_gap: "Cost estimate gap", spec_ambiguity: "Unclear specifications", agency_capacity: "Agency capacity",
  market_thin: "Thin market / few bidders", gov_delay: "Approval delays", supplier_performance: "Contractor performance" };
const pick = (list, ...prefs) => prefs.find((p) => list.includes(p)) || list[0];

function PackageForm({ meta, onRun, busy }) {
  const o = meta.options;
  const [f, setF] = useState(() => ({
    description: "Rehabilitation of rural roads", category: pick(o.categories, "Civil Works"),
    sector: pick(o.sectors, "Transportation", "Transport"), country_code: pick(o.countries.map((c) => c.code), "KE"),
    value_usd_m: 20, method: pick(o.methods, "Request for Bids"), review_type: pick(o.review_types, "Prior"),
  }));
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const submit = (e) => { e.preventDefault(); onRun({ ...f, value_usd_m: f.value_usd_m === "" ? null : Number(f.value_usd_m) }); };
  return (
    <form className="card sticky" onSubmit={submit}>
      <h2>Your package</h2>
      <label className="field"><span>Short description</span><input value={f.description} onChange={set("description")} maxLength={300} /></label>
      <label className="field"><span>Procurement category</span><select value={f.category} onChange={set("category")}>{o.categories.map((x) => <option key={x}>{x}</option>)}</select></label>
      <label className="field"><span>Sector</span><select value={f.sector} onChange={set("sector")}>{o.sectors.map((x) => <option key={x}>{x}</option>)}</select></label>
      <div className="fields-2">
        <label className="field"><span>Country</span><select value={f.country_code} onChange={set("country_code")}>{o.countries.map((c) => <option key={c.code} value={c.code}>{c.name}</option>)}</select></label>
        <label className="field"><span>Value (USD m)</span><input type="number" min="0" step="any" value={f.value_usd_m} onChange={set("value_usd_m")} /></label>
      </div>
      <label className="field"><span>Procurement method you are considering</span><select value={f.method} onChange={set("method")}>{o.methods.map((x) => <option key={x}>{x}</option>)}</select></label>
      <button className="btn primary block" style={{ marginTop: 16 }} disabled={busy}>{busy ? <Spinner label="Analysing…" /> : "Analyse package"}</button>
      <p className="tiny" style={{ marginTop: 8 }}>Uses {meta.summary.contracts.toLocaleString()} past contracts and {meta.summary.documents} project reports.</p>
    </form>
  );
}

function Schedule({ r }) {
  const f = r.facts, s = r.schedule, m = r.market;
  return (
    <div className="stack">
      <div className="card">
        <h2>How long to award?</h2>
        <RangeBar p10={f.model_p10_days} p50={f.model_p50_days} p90={f.model_p90_days} />
        <div className="stats" style={{ marginTop: 14 }}>
          <Stat label="Plan to sign the contract in" value={`about ${dur(s.plan_days)}`} sub="the usual time for a tender like this" />
          <Stat label="Allow up to" value={dur(s.contingency_days)} sub="9 in 10 similar tenders are signed by then" />
          <Stat label="Chance of re-tendering" value={`${f.model_rebid_pct}%`} sub={s.add_rebid_buffer ? "high: leave time to advertise it again" : "chance it has to be advertised again"} />
          <Stat label="Similar past contracts" value={`about ${dur(f.bench_median_days)}`} sub={`typical of ${f.n_comparable} signed contracts`} />
        </div>
        <p className="tiny" style={{ marginTop: 10 }}>
          Forecast from a survival model trained on signed contracts and on tenders still open, so slow tenders are not left out.
          Similar past contracts include only tenders that already signed, so they tend to look faster.
        </p>
      </div>
      <div className="card">
        <h2>Market approach options</h2>
        <p className="small muted">The same package under each procurement method used for {r.spec.category.toLowerCase()} in the data.</p>
        {/* Two forecast columns named after what to do with them (as in the card above) + the evidence behind each row.
            No re-tender column: too few real re-tenders in the data to measure it. A method no past contract of this
            category used gets no forecast: the model would only be extrapolating. */}
        <table className="tbl">
          <thead><tr><th>Method</th><th className="num">Plan for</th><th className="num">Allow up to</th><th>Based on</th></tr></thead>
          <tbody>{m.methods.map((x) => {
            const none = !x.past_contracts;
            return (
              <tr key={x.method}><td>{x.method} <span className="row" style={{ display: "inline-flex", gap: 4 }}>
                {x.chosen && <span className="tag accent">your choice</span>}{x.fastest && <Status level="good">fastest</Status>}</span></td>
                <td className="num" title={none ? "" : `${x.p50_days} days`}>{none ? "–" : dur(x.p50_days)}</td>
                <td className="num" title={none ? "" : `${x.p90_days} days`}>{none ? "–" : dur(x.p90_days)}</td>
                <td className="small">{none ? <span className="muted">No {r.spec.category.toLowerCase()} contract has used this method</span>
                  : <span title={x.past_median_days != null ? `These contracts usually took ${dur(x.past_median_days)}. Only signed contracts are counted, so this looks shorter than the forecast.` : ""}>
                    {x.past_contracts} past contract{x.past_contracts === 1 ? "" : "s"}
                    {x.thin && <span className="row" style={{ display: "inline-flex", marginLeft: 6 }}><Status level="warning">few: a rough guide only</Status></span>}</span>}</td></tr>);
          })}</tbody>
        </table>
      </div>
    </div>
  );
}

function Risks({ r }) {
  const risks = r.findings.filter((x) => x.kind === "risk");
  return (
    <div className="stack">
      <div className="card">
        <h2>Risks from past projects ({risks.length})</h2>
        {!risks.length && <Empty title="No grounded precedent found">Tenderwise only shows a risk when a past report supports it. Try the Ask tab with your own wording.</Empty>}
        {risks.map((x, i) => (
          <div key={i} className="removed" style={{ paddingTop: 12 }}>
            <div className="row"><span className="tag accent">{THEME[x.theme] || x.theme}</span>
              {x.evidence[0]?.scope && <span className="tiny">{x.evidence[0].scope}</span>}{x.caution && <Status level="warning">ratings disagree on the cited project</Status>}</div>
            <blockquote className="quote">“{x.claim}”</blockquote>
            <p className="small" style={{ margin: "6px 0" }}><b>Mitigation:</b> {x.recommendation}
              <span className="tiny"> · {x.mitigation_source === "from the cited lesson" ? "written from this lesson and checked against it" : "standard practice for this risk"}</span></p>
            <div className="spread">
              <span className="row">{x.evidence.map((e) => <button key={e.chunk_id} className="btn ghost small" onClick={() => openPassage(e.chunk_id)}>
                Lesson: {e.doc_id} · page {e.page} →</button>)}</span>
              <Feedback ids={x.evidence.map((e) => e.chunk_id)} question={r.spec.description} surface="package" />
            </div>
          </div>
        ))}
      </div>
      {r.conflicts.length > 0 && <div className="card"><h2>Where ratings disagree on cited projects</h2>
        {r.conflicts.map((c) => <p key={c.project_id} className="small" title={`rule ${c.rule_applied}; ${c.note}`}><b>{c.project_id}</b>: the last status report rated progress {c.isr_last.rating}; the completion report rated the outcome {c.icr.rating}. The completion report's rating is used, as the final assessment. {c.note.split(";")[0]}.</p>)}</div>}
      <details className="card"><summary>Checks applied ({r.findings.length} statements passed, {r.rejected.length} rejected)</summary>
        {r.rejected.map((x, i) => <div key={i} className="removed"><s>{x.claim}</s><div className="err">{(x.rejected_because || []).join("; ")}</div></div>)}
        <ol className="tiny">{r.trace.map((t, i) => <li key={i}>{t}</li>)}</ol></details>
    </div>
  );
}

export default function PackageWorkspace({ meta, pkg, setPkg }) {
  const [r, setR] = useState(pkg?.report || null);
  const [step, setStep] = useState(1), [busy, setBusy] = useState(false), [err, setErr] = useState(null);
  const run = async (form) => {
    setBusy(true); setErr(null);
    try { const rep = await api.analyze(form); setR(rep); setPkg({ ...rep.spec, report: rep }); setStep(1); } catch (e) { setErr(e.message); }
    setBusy(false);
  };
  const title = useMemo(() => r && `${r.spec.category} · ${r.spec.sector} · ${r.spec.country}${r.spec.value_usd_m ? ` · USD ${r.spec.value_usd_m}m` : ""}`, [r]);
  return (
    <>
      <div className="page-head"><h1>Package workspace</h1>
        <p>Plan a package from evidence: schedule, market approach and risks.</p></div>
      <div className="grid-2">
        <PackageForm meta={meta} onRun={run} busy={busy} />
        <div>
          <ErrorBox error={err} />
          {!r && !busy && <div className="card"><Empty title="Describe your package and press Analyse">
            You will get a schedule range, a comparison of procurement methods, and risks from past projects with mitigations.</Empty></div>}
          {busy && !r && <div className="card"><Skeleton lines={6} /></div>}
          {r && <>
            <div className="card" style={{ marginBottom: 14 }}>
              <div className="spread"><div><div className="tiny">Analysed package</div><b>{title}</b><div className="small muted">{r.spec.method}</div></div>
                <button className="btn" onClick={() => go("/ask?from=package")}>💬 Ask about this package</button></div>
            </div>
            <div className="steps" role="tablist">
              {["Schedule & market approach", "Risks & mitigations"].map((l, i) =>
                <button key={l} role="tab" aria-selected={step === i + 1} className={step === i + 1 ? "on" : ""} onClick={() => setStep(i + 1)}><span className="n">{i + 1}</span>{l}</button>)}
            </div>
            {step === 1 && <Schedule r={r} />}
            {step === 2 && <Risks r={r} />}
          </>}
        </div>
      </div>
    </>
  );
}
