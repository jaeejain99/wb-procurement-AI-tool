"""Execution & monitoring: early-warning indicators for active projects/contracts.

Each warning comes in two forms: `flags` (exact, technical: what the agent cites and the verifier checks numbers against) and
`alerts` (the same fact in plain words for the screens: a headline, a detail sentence, an optional caveat)."""
from __future__ import annotations
import re
import pandas as pd

RATING = {6: "Highly Satisfactory", 5: "Satisfactory", 4: "Moderately Satisfactory",
          3: "Moderately Unsatisfactory", 2: "Unsatisfactory", 1: "Highly Unsatisfactory"}


def plain_duration(days) -> str:
    """43 -> '43 days'; 547 -> '18 months'; 900 -> '2.5 years'. The UI uses the same rule (web/src/plain.js)."""
    d = max(0, int(round(float(days))))
    if d < 60:
        return f"{d} day{'' if d == 1 else 's'}"
    months = d / 30.44
    if months < 23.5:
        return f"{round(months)} months"
    y = round(months / 12 * 2) / 2
    return f"{y:g} years"


def grade_gap(gap: float) -> str:
    """ISR-vs-ICR gap on the 1-6 rating scale in words."""
    a = abs(gap)
    size = "slightly" if a < 0.35 else "about half a grade" if a < 0.75 else "about one grade" if a < 1.25 else f"about {round(a)} grades"
    return f"{size} {'higher' if gap > 0 else 'lower'}"


def _rating(x) -> str:
    return RATING.get(int(round(min(6, max(1, float(x))))), str(x))


def _date(ts) -> str:
    t = pd.Timestamp(ts)
    return f"{t.day} {t.strftime('%B %Y')}"


def as_of_date(plat) -> pd.Timestamp:
    """'Today' for the data: the latest pull (open tenders' censoring date), else the latest award or ISR."""
    ds = plat.ds
    cands = [pd.to_datetime(ds.contracts.award_date).max()]
    o = getattr(ds, "open_tenders", None)
    if o is not None and len(o):
        cands.append(pd.to_datetime(o.censored_at).max())
    if len(ds.isr):
        cands.append(pd.to_datetime(ds.isr.date).max())
    return max(d for d in cands if pd.notna(d)).normalize()


def project_warnings(plat, pid: str, as_of: str) -> dict:
    as_of = pd.Timestamp(as_of)
    ds, flags, alerts, score = plat.ds, [], [], 0.0
    isr = ds.isr[(ds.isr.project_id == pid) & (ds.isr.date <= as_of)].sort_values("date")
    bias = plat.bias()["mean_gap"]
    if len(isr) >= 3 and isr.dev_objective_rating.iloc[-1] <= isr.dev_objective_rating.iloc[-3] - 1:
        flags.append("ISR development-objective rating fell >=1 notch over the last two reports"); score += .3
        alerts.append(dict(kind="rating_drop", level="serious", title="Progress rating dropped",
                           detail=f"From {_rating(isr.dev_objective_rating.iloc[-3])} to {_rating(isr.dev_objective_rating.iloc[-1])} "
                                  f"over the last two status reports (latest {_date(isr.date.iloc[-1])}).", tech=flags[-1]))
    if len(isr):
        last = isr.dev_objective_rating.iloc[-1]
        adj = last - bias
        if adj <= 3.5:
            flags.append(f"bias-adjusted ISR rating {adj:.1f} (raw {last}, portfolio ISR optimism {bias:+.2f}) is at/below Moderately Unsatisfactory"); score += .3
            if abs(bias) >= 0.35 and last > 3:
                alerts.append(dict(kind="adjusted_rating", level="serious", title="Likely weaker than reported",
                                   detail=f"The latest status report rates it {_rating(last)}. Status reports in this portfolio rate projects "
                                          f"{grade_gap(bias)} than the final completion reports, which puts it closer to {_rating(min(adj, 3))}.",
                                   tech=flags[-1]))
            else:
                alerts.append(dict(kind="adjusted_rating", level="serious", title="Low progress rating",
                                   detail=f"The latest status report rates it {_rating(last)}.", tech=flags[-1]))
    # disbursement vs historical distribution of closed projects at the same project age
    closed = set(ds.projects[ds.projects.status == "Closed"].project_id)
    d = ds.disbursements[(ds.disbursements.project_id == pid) & (ds.disbursements.date <= as_of)].sort_values("date")
    if len(d):
        age, cur = int(d.project_age_years.iloc[-1]), float(d.cum_disbursed_pct.iloc[-1])
        hist = ds.disbursements[ds.disbursements.project_id.isin(closed) & (ds.disbursements.project_age_years == age)].cum_disbursed_pct
        if len(hist) >= 10 and cur < hist.quantile(.25):
            flags.append(f"disbursement {cur:.0f}% at year {age} is below the historical P25 ({hist.quantile(.25):.0f}%)"); score += .25
            alerts.append(dict(kind="disbursement", level="warning", title="Spending is slow",
                               detail=f"{cur:.0f}% of the funds used after {age} year{'' if age == 1 else 's'}; 3 in 4 similar closed projects "
                                      f"had used more than {hist.quantile(.25):.0f}% by then.", tech=flags[-1]))
    # tenders still open past their predicted P90: the real open tenders (noticed, no award yet), plus - in a back-dated view -
    # contracts that signed only after as_of
    rows = overdue_tenders(plat, pid, as_of)
    for t in rows:
        if t["overdue"]:
            flags.append(f"tender {t['tender_id']} ({t['category']}) open {t['days_open']}d > predicted P90 {t['p90_days']}d"); score += .25
            what = " ".join(t["description"].split()).rstrip(".") if t["description"].strip() else f"A {str(t['category'] or 'procurement').lower()} tender"
            what = what[0].upper() + what[1:].lower() if what.isupper() else what[0].upper() + what[1:]
            opened, usual = plain_duration(t["days_open"]), plain_duration(t["p90_days"])
            if opened == usual:                                   # 365 vs 359 days: both '12 months' -> say the days
                opened, usual = f"{opened} ({t['days_open']} days)", f"{usual} ({t['p90_days']} days)"
            if len(what) > 70:                                    # cut at a word, not mid-word
                what = what[:70].rsplit(" ", 1)[0].rstrip(" ,;:-(") + "…"
            alerts.append(dict(kind="tender", level="critical", title=f"{what} is taking much longer than usual",
                               detail=f"Open for {opened} (since {_date(t['notice_date'])}). 9 in 10 similar tenders "
                                      f"are signed within {usual}.",
                               note=("No signed contract found in World Bank records yet. It may be signed but not published, so check "
                                     "its status with the project team.") if t["source"] == "notice" else None,
                               tender_id=t["tender_id"], tech=flags[-1]))
    # packages behind their OWN Procurement Plan: planned signing passed > 30 days ago, plan shows no signature (recent plans only)
    late = behind_plan(plan_of(plat, pid), as_of, skip_activities={t.get("step_ref") and _act(t["step_ref"]) for t in rows if t["overdue"]})
    if late:
        worst = late[0]
        flags.append(f"{len(late)} plan activit{'y' if len(late) == 1 else 'ies'} past planned signing; worst {worst['step_ref']} planned "
                     f"{worst['signed_planned']} ({worst['days_late']}d late, plan of {worst['plan_date']})"); score += .25
        what = (worst["description"] or worst["step_ref"])
        what = what if len(what) <= 70 else what[:70].rsplit(" ", 1)[0] + "…"
        alerts.append(dict(kind="plan", level="serious",
                           title=f"{len(late)} package{'s are' if len(late) > 1 else ' is'} behind {'their' if len(late) > 1 else 'its'} own plan",
                           detail=(f"Most delayed: {what} — contract signing was planned for {_date(worst['signed_planned'])}, "
                                   f"{plain_duration(worst['days_late'])} ago, and the plan does not show it signed."),
                           note=f"From the project's Procurement Plan published {_date(worst['plan_date'])}.", tech=flags[-1],
                           packages=[dict(step_ref=x["step_ref"], description=x["description"], signed_planned=x["signed_planned"],
                                          days_late=x["days_late"], status=x["status"]) for x in late[:10]]))
    return dict(project_id=pid, as_of=str(as_of.date()), risk_score=round(min(score, 1.0), 2), flags=flags, alerts=alerts,
                open_tenders=len(rows), overdue_tenders=sum(t["overdue"] for t in rows), behind_plan=len(late))


def _act(ref):
    import re
    m = re.match(r"([A-Z]{2})-(?:[A-Z0-9&.()]+-)*?(\d{4,8})-(?:CW|GO|CS|NC)", str(ref or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def plan_of(plat, pid) -> pd.DataFrame:
    p = getattr(plat.ds, "plan", None)
    return p[p.project_id == pid] if p is not None and len(p) else pd.DataFrame()


def tidy_description(text) -> str:
    """Plan cells wrap mid-word ('E ngagement', 'Desi gn'): re-join the obvious fragments and drop loan numbers ('IDA / 71320')."""
    import re
    t = re.sub(r"\b(?:IDA|IBRD|TF|COFN|GA)\s*/\s*[A-Z0-9-]+", "", str(text or "")).strip(" /-")
    t = re.sub(r"\b([A-Z]) ([a-z]{2,})", r"\1\2", t)                        # 'E ngagement' -> 'Engagement'
    short = ("of", "to", "in", "on", "at", "an", "or", "by", "a", "as", "is", "be", "we", "it")
    t = re.sub(r"\b([A-Za-z]{2,}) ([a-z]{1,2})\b", lambda m: m.group(0) if m.group(2) in short else m.group(1) + m.group(2), t)
    return re.sub(r"\s+", " ", t).strip()[:200]


IN_PROGRESS = ("pending implementation", "under implementation", "under review", "draft", "not started")


def behind_plan(plan: pd.DataFrame, as_of, grace_days=30, max_plan_age_days=365, max_days_late=730, skip_activities=()) -> list[dict]:
    """Plan activities whose planned signing date passed more than `grace_days` ago with no actual signing date and a status that
    is not signed / completed / cancelled. Only plans published within `max_plan_age_days` of as_of (an old plan's dates are
    stale). Most delayed first."""
    if plan is None or not len(plan):
        return []
    as_of = pd.Timestamp(as_of)
    # only packages the plan explicitly shows as still in progress (an unparsed status is not evidence of a delay)
    p = plan[(pd.to_datetime(plan.plan_date) >= as_of - pd.Timedelta(days=max_plan_age_days)) & plan.signed_planned.notna()
             & plan.signed_actual.isna() & plan.status.fillna("").str.lower().isin(IN_PROGRESS)]
    late_days = (as_of - pd.to_datetime(p.signed_planned)).dt.days
    # > 2 years past the planned date and still 'pending' is an entry nobody updated, not an early warning
    p = p[(late_days > grace_days) & (late_days <= max_days_late)]
    # a reference wrapped across two lines in the plan PDF ('.../082' + '83 / ...') is parsed twice: drop the truncated copy
    # when the full reference ('.../082/83') with the same planned signing date is also there
    refs = {(str(r.step_ref), str(r.signed_planned)) for r in p.itertuples()}
    trunc = lambda r: any(f.startswith(str(r.step_ref) + "/") and d == str(r.signed_planned) for f, d in refs)
    out = []
    for r in p.itertuples():
        if getattr(r, "step_activity", None) in skip_activities or trunc(r):
            continue
        desc = re.sub(r"^\s*\d{1,3}\s*/\s*", "", str(r.description or ""))      # the wrapped tail of the reference, e.g. '83 / '
        out.append(dict(step_ref=r.step_ref, description=tidy_description(desc), status=r.status,
                        signed_planned=str(pd.Timestamp(r.signed_planned).date()), plan_date=str(pd.Timestamp(r.plan_date).date()),
                        days_late=int((as_of - pd.Timestamp(r.signed_planned)).days)))
    return sorted(out, key=lambda x: -x["days_late"])


def overdue_tenders(plat, pid: str, as_of) -> list[dict]:
    """Each open tender of a project with days open, its predicted P50/P90 and whether it is past the P90."""
    as_of, ds = pd.Timestamp(as_of), plat.ds
    parts = []
    o = getattr(ds, "open_tenders", None)
    if o is not None and len(o):
        o = o[(o.project_id == pid) & (o.notice_date <= as_of)]
        if len(o):
            parts.append(o.assign(tender_id=o.tender_id.astype(str), source="notice"))
    c = ds.contracts[(ds.contracts.project_id == pid) & (ds.contracts.notice_date <= as_of) & (ds.contracts.award_date > as_of)
                     & ds.contracts.tender_days.notna()]
    if len(c):
        parts.append(c.assign(tender_id=c.contract_id, description=c.get("description", ""), source="signed_later"))
    if not parts:
        return []
    t = pd.concat(parts, ignore_index=True)
    pr = plat.models.predict_df(t)
    out = []
    for (_, r), (_, q) in zip(t.iterrows(), pr.iterrows()):
        days = int((as_of - r.notice_date).days)
        deadline = r.get("submission_deadline")
        advertising = bool(pd.notna(deadline) and pd.Timestamp(deadline) >= as_of)     # bids still being received: not late
        ref = r.get("step_ref")
        out.append(dict(tender_id=r.tender_id, category=r.get("category"), method=r.get("procurement_method"),
                        description=str(r.get("description", "") or "")[:160], notice_date=str(r.notice_date.date()), days_open=days,
                        p50_days=int(round(q.p50_days)), p90_days=int(round(q.p90_days)),
                        overdue=bool(days > q.p90_days and not advertising), advertising=advertising,
                        submission_deadline=str(pd.Timestamp(deadline).date()) if pd.notna(deadline) else None,
                        step_ref=str(ref) if isinstance(ref, str) and ref.strip() else None, source=r.get("source", "notice")))
    return sorted(out, key=lambda x: -x["days_open"])
