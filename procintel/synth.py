"""Synthetic dataset that mimics the *shape* of the World Bank sources named in the assessment
(Major Contract Awards, Projects & Operations, WGI/CPIA, disbursements, ISRs, ICRs).

Causal structure is planted (governance -> slower tenders, big ICB works -> re-bids, ...) so the models
and retrieval have real signal to find. Ground-truth theme labels for evaluation are returned separately and
are never visible to the pipeline."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .themes import THEMES

COUNTRIES = {  # code: (name, region, wgi_rule_of_law, wgi_gov_effectiveness, wgi_reg_quality, cpia)
    "GHA": ("Ghana", "Sub-Saharan Africa", 0.0, -0.1, 0.0, 3.8),
    "KEN": ("Kenya", "Sub-Saharan Africa", -0.4, -0.3, -0.2, 3.7),
    "NPL": ("Nepal", "South Asia", -0.5, -0.7, -0.6, 3.6),
    "BGD": ("Bangladesh", "South Asia", -0.7, -0.7, -0.8, 3.3),
    "VNM": ("Vietnam", "East Asia", 0.0, 0.0, -0.4, 3.9),
    "PER": ("Peru", "Latin America", -0.6, -0.2, 0.4, 3.9),
    "MOZ": ("Mozambique", "Sub-Saharan Africa", -0.9, -0.8, -0.7, 3.1),
    "JOR": ("Jordan", "Middle East", 0.2, 0.0, 0.0, 3.7),
}
SECTORS = ["Transport", "Water", "Energy", "Health", "Education", "Agriculture"]
CATEGORIES = ["Civil Works", "Goods", "Consultancies", "Non-Consulting Services"]
RATING = {6: "Highly Satisfactory", 5: "Satisfactory", 4: "Moderately Satisfactory",
          3: "Moderately Unsatisfactory", 2: "Unsatisfactory", 1: "Highly Unsatisfactory"}
SUPPLIERS = ["Acme Construction", "Baraka Engineering", "Delta Infrastructure", "Sunrise Medical Supplies",
             "Nile Water Systems", "Orion Power Solutions", "Pacific Roadworks", "Zenith Consulting Group",
             "Andes Civil Contractors", "Kilima Builders", "Meridian Agri Equipment", "Lotus ICT Services",
             "Summit Electrical", "Harbor Marine Works", "Savanna Logistics", "Everest Design Associates"]
SUFFIXES = ["Ltd", "Ltd.", "Limited", "LTD", "Co. Ltd", "(Pvt) Ltd", "Inc", "S.A."]
BASE_DAYS = {"Civil Works": 150, "Goods": 110, "Consultancies": 120, "Non-Consulting Services": 90}
METHOD_MULT = {"ICB": 1.35, "NCB": 1.0, "RFQ": 0.6, "QCBS": 1.2, "CQS": 0.8, "ICS": 0.9, "Direct": 0.5}
WORKS = {"Transport": ["Rehabilitation of {n} km of rural roads", "Construction of a {n}-span bridge"],
         "Water": ["Construction of a {n} MLD water treatment plant", "Rehabilitation of {n} km of water distribution network"],
         "Energy": ["Construction of {n} km of 33kV transmission line", "Installation of a {n} MW solar plant"],
         "Health": ["Construction of {n} district hospitals", "Rehabilitation of {n} primary health clinics"],
         "Education": ["Construction of {n} secondary schools", "Rehabilitation of {n} classroom blocks"],
         "Agriculture": ["Construction of {n} irrigation schemes", "Rehabilitation of {n} km of irrigation canals"]}
GOODS = {"Transport": "road maintenance equipment", "Water": "pumps and pipes", "Energy": "transformers and meters",
         "Health": "medical equipment", "Education": "school furniture and ICT equipment", "Agriculture": "tractors and farm inputs"}


@dataclass
class SynthData:
    projects: pd.DataFrame
    contracts: pd.DataFrame
    governance: pd.DataFrame
    disbursements: pd.DataFrame
    isr: pd.DataFrame
    docs: list                       # [{doc_id, doc_type, project_id, filename, text}]
    eval_labels: list = field(default_factory=list)   # [{doc_id, theme, sentence}]  (ground truth, hidden)


def _supplier_variant(rng, base):
    name = base.replace("Construction", "Constr.") if rng.random() < .15 else base
    name = name + " " + str(rng.choice(SUFFIXES))
    return name.upper() if rng.random() < .15 else name


def make_governance(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for code, (name, region, rl, ge, rq, cpia) in COUNTRIES.items():
        for y in range(2010, 2026):
            d = 0.015 * (y - 2010)
            rows.append(dict(country_code=code, country=name, region=region, year=y,
                             available_from=pd.Timestamp(y + 1, 9, 30),   # WGI published ~9 months after year-end
                             wgi_rule_of_law=round(rl + d * np.sign(rl + .01) + rng.normal(0, .05), 3),
                             wgi_gov_effectiveness=round(ge + d + rng.normal(0, .05), 3),
                             wgi_reg_quality=round(rq + d + rng.normal(0, .05), 3),
                             cpia=round(cpia + rng.normal(0, .08), 2)))
    return pd.DataFrame(rows)


def _gov_asof(gov, code, date):
    g = gov[(gov.country_code == code) & (gov.available_from <= date)]
    return g.iloc[-1] if len(g) else gov[gov.country_code == code].iloc[0]


def generate(seed=7, n_projects=110, id_start=150000, approval_years=(2012, 2022), cut="2025-10-31",
             duration_shift=1.0, gov=None) -> SynthData:
    rng = np.random.default_rng(seed)
    cut = pd.Timestamp(cut)
    gov = make_governance() if gov is None else gov
    projects, contracts, isr_rows, disb_rows, root = [], [], [], [], {}
    for i in range(n_projects):
        pid = f"P{id_start + i:06d}"
        code = str(rng.choice(list(COUNTRIES)))
        sector = str(rng.choice(SECTORS))
        appr = pd.Timestamp(int(rng.integers(*approval_years)), int(rng.integers(1, 13)), 15)
        closing = appr + pd.Timedelta(days=365 * int(rng.integers(5, 8)))
        status = "Closed" if closing <= cut else "Active"
        es = str(rng.choice(["High", "Substantial", "Moderate", "Low"], p=[.1, .3, .4, .2]))
        projects.append(dict(project_id=pid, project_name=f"{COUNTRIES[code][0]} {sector} Improvement Project {i}",
                             country_code=code, sector=sector, commitment_usd_m=round(float(rng.lognormal(4.1, .6)), 1),
                             approval_date=appr, closing_date=closing, status=status, es_risk=es,
                             implementing_agency=f"{COUNTRIES[code][0]} Ministry of {sector}"))
        span = (closing - appr).days
        for k in range(int(rng.integers(3, 8))):
            hi = min(span - 600, (cut - appr).days - 400)
            if hi <= 90:
                continue
            notice = appr + pd.Timedelta(days=int(rng.integers(60, hi)))
            cat = str(rng.choice(CATEGORIES, p=[.45, .25, .2, .1]))
            mu, sd = {"Civil Works": (2.1, .9), "Goods": (.9, .8), "Consultancies": (-.2, .7), "Non-Consulting Services": (-.5, .6)}[cat]
            value = round(float(rng.lognormal(mu, sd)), 2)
            if cat in ("Consultancies",):
                method = str(rng.choice(["QCBS", "CQS", "ICS"], p=[.6, .25, .15]))
            else:
                method = "ICB" if value > 5 else str(rng.choice(["NCB", "RFQ", "ICB"], p=[.55, .35, .1]))
            review = "Prior" if value > 8 else "Post"
            g = _gov_asof(gov, code, notice)
            ge, rl = g.wgi_gov_effectiveness, g.wgi_rule_of_law
            exp_days = (BASE_DAYS[cat] * METHOD_MULT[method] * (1.25 if review == "Prior" else 1) *
                        np.exp(-0.30 * ge) * (1 + .15 * np.log1p(value)))
            logit = -2.6 + .9 * (-ge) + .6 * (method == "ICB") + .35 * np.log1p(value) + .5 * (cat == "Civil Works")
            rebid = bool(rng.random() < 1 / (1 + np.exp(-logit)))
            days = exp_days * float(rng.lognormal(0, .18)) + (rng.integers(90, 150) if rebid else 0)
            days *= duration_shift
            lam = max(.1, .25 + .5 * (-rl) + .12 * np.log1p(value) + .2 * rebid)
            amend = int(rng.poisson(lam))
            n = int(rng.integers(2, 120))
            if cat == "Civil Works":
                desc = str(rng.choice(WORKS[sector])).format(n=n)
            elif cat == "Goods":
                desc = f"Supply of {GOODS[sector]} for {sector.lower()} programme"
            elif cat == "Consultancies":
                desc = str(rng.choice([f"Design and supervision consultancy for {sector.lower()} works",
                                       f"Feasibility and safeguards study for {sector.lower()} investments"]))
            else:
                desc = f"Operation, maintenance and data collection services for {sector.lower()} facilities"
            cid = f"{pid}-C{k:02d}"
            sup = _supplier_variant(rng, str(rng.choice(SUPPLIERS)))
            # hidden root cause (drives the narrative text in the ICR, not available as a feature)
            if rebid:
                rc = str(rng.choice(["estimate_gap", "spec_ambiguity"], p=[.55, .45]))
            elif days > 1.3 * exp_days:
                rc = str(rng.choice(["gov_delay", "agency_capacity", "market_thin"]))
            elif amend >= 1 and rng.random() < .6:
                rc = "spec_ambiguity"
            elif rng.random() < .12:
                rc = "supplier_performance"
            else:
                rc = None
            root[cid] = (rc, desc, sup, days, exp_days)
            contracts.append(dict(contract_id=cid, project_id=pid, description=desc, category=cat,
                                  procurement_method=method, review_type=review, value_usd_m=value,
                                  supplier_raw=sup, country_code=code, sector=sector, notice_date=notice,
                                  award_date=notice + pd.Timedelta(days=int(days)), amendments=amend, rebid=int(rebid)))
        # ISR trajectory + ICR rating (planted optimism bias in ISRs)
        r, d, ratings = 4.4 + rng.normal(0, .5), appr + pd.Timedelta(days=365), []
        end = min(closing, cut)
        while d <= end:
            r = float(np.clip(r + rng.normal(0, .35) - (.1 if COUNTRIES[code][3] < -.5 else 0), 1.5, 5.8))
            ratings.append((d, int(round(r))))
            d += pd.Timedelta(days=182)
        for j, (dt, rt) in enumerate(ratings):
            isr_rows.append(dict(isr_id=f"{pid}-ISR{j+1:02d}", project_id=pid, date=dt, dev_objective_rating=rt,
                                 implementation_progress_rating=int(np.clip(rt + rng.integers(-1, 2), 1, 6))))
        # disbursement as cumulative fraction by project age (S-curve, slower for weak governance)
        speed = 1.0 + .25 * COUNTRIES[code][3] + rng.normal(0, .1)
        for age in range(1, 8):
            dt = appr + pd.Timedelta(days=365 * age)
            if dt > cut:
                break
            frac = 1 / (1 + np.exp(-(age - 3.2) * speed))
            disb_rows.append(dict(project_id=pid, date=dt, project_age_years=age,
                                  cum_disbursed_pct=round(float(np.clip(frac + rng.normal(0, .03), 0, 1)) * 100, 1)))
    projects = pd.DataFrame(projects)
    contracts = pd.DataFrame(contracts)
    isr = pd.DataFrame(isr_rows)
    # missing data, as in real registries
    m = rng.random(len(contracts))
    contracts.loc[m < .07, "value_usd_m"] = np.nan
    contracts.loc[(m > .07) & (m < .12), "award_date"] = pd.NaT
    # documents
    docs, labels = [], []
    pmap = projects.set_index("project_id")
    for pid, p in pmap.iterrows():
        pisr = isr[isr.project_id == pid].sort_values("date")
        if not len(pisr):
            continue
        for _, row in pisr.iterrows():
            did = row.isr_id
            txt = (f"# Implementation Status and Results Report {did}\n\nCountry: {COUNTRIES[p.country_code][0]}\nSector: {p.sector}\n"
                   f"Project: {pid}\nDate: {row.date.date()}\n\n## 1. Ratings\n\nProgress towards the development objective is rated "
                   f"{RATING[row.dev_objective_rating]}. Overall implementation progress is rated {RATING[row.implementation_progress_rating]}.\n\n"
                   f"## 2. Procurement\n\nProcurement is proceeding broadly as planned; the implementing agency has submitted updated procurement plans.\n")
            docs.append(dict(doc_id=did, doc_type="ISR", project_id=pid, filename=f"{did}.md", text=txt))
        if p.status != "Closed":
            continue
        last = int(pisr.dev_objective_rating.iloc[-1])
        bias = int(rng.choice([0, 1, 2, 3], p=[.5, .25, .15, .1]))
        icr_rating = int(np.clip(last - bias, 1, 6))
        pc = contracts[contracts.project_id == pid]
        issues, lessons, seen = [], [], set()
        for _, c in pc.iterrows():
            rc, desc, sup, days, exp_days = root[c.contract_id]
            if rc is None or rng.random() < .1:
                continue
            t = THEMES[rc]
            fill = dict(desc=desc, pct=int(rng.integers(15, 60)), months=int(rng.integers(3, 12)), n=int(rng.integers(2, 6)),
                        weeks=int(rng.integers(3, 10)), supplier=sup, country=COUNTRIES[p.country_code][0])
            s = str(rng.choice(t["issue"])).format(**fill)
            issues.append(f"Contract {c.contract_id}: {s}")
            labels.append(dict(doc_id=f"ICR_{pid}", theme=rc, sentence=s))
            if rc not in seen:
                seen.add(rc)
                ls = str(rng.choice(t["lesson"]))
                lessons.append(ls)
                labels.append(dict(doc_id=f"ICR_{pid}", theme=rc, sentence=ls))
        rows = "\n".join(f"| {c.contract_id} | {c.category} | {c.procurement_method} | "
                         f"{'n/a' if pd.isna(c.value_usd_m) else c.value_usd_m} | {c.supplier_raw} |" for _, c in pc.iterrows())
        txt = (f"# Implementation Completion and Results Report: {p.project_name} ({pid})\n\n"
               f"Country: {COUNTRIES[p.country_code][0]}\nSector: {p.sector}\nProject: {pid}\nDate: {p.closing_date.date()}\n\n"
               f"## 1. Project Context and Development Objectives\n\nThe project supported {p.sector.lower()} sector reforms and investments in "
               f"{COUNTRIES[p.country_code][0]} with a commitment of USD {p.commitment_usd_m} million, implemented by the {p.implementing_agency}.\n\n"
               f"## 2. Outcome Ratings\n\n| Dimension | Rating |\n| --- | --- |\n| Overall Outcome | {RATING[icr_rating]} |\n\n"
               f"## 3. Procurement Performance\n\n" + ("\n\n".join(issues) if issues else "Procurement proceeded without significant issues.") +
               f"\n\n## 4. Lessons Learned\n\n" + ("\n".join(f"- {l}" for l in lessons) if lessons else "- No procurement-specific lessons were recorded.") +
               f"\n\n## Annex 2. Contract Summary\n\n| Contract | Category | Method | Value (USD m) | Supplier |\n| --- | --- | --- | --- | --- |\n{rows}\n")
        docs.append(dict(doc_id=f"ICR_{pid}", doc_type="ICR", project_id=pid, filename=f"ICR_{pid}.md", text=txt))
    return SynthData(projects, contracts, make_governance() if gov is None else gov, pd.DataFrame(disb_rows), isr, docs, labels)


def write_dataset(d: SynthData, out):
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    d.contracts.to_csv(os.path.join(out, "contracts.csv"), index=False)
    d.governance.to_csv(os.path.join(out, "governance.csv"), index=False)
    d.disbursements.to_csv(os.path.join(out, "disbursements.csv"), index=False)
    d.projects.to_json(os.path.join(out, "projects.json"), orient="records", date_format="iso", indent=1)
    d.isr.to_json(os.path.join(out, "isr.json"), orient="records", date_format="iso", indent=1)
    for doc in d.docs:
        with open(os.path.join(out, "docs", doc["filename"]), "w") as f:
            f.write(doc["text"])
    # ground truth for evaluation only
    with open(os.path.join(out, "_eval_labels.json"), "w") as f:
        json.dump(d.eval_labels, f)
