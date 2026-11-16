"""Platform: wires ingestion, index, graph, models and the copilot together."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .graph import KnowledgeGraph
from .guardrails import RATING_NUM, optimism_bias
from .index import HybridIndex
from .live_wb import parse_icr_rating
from .ingest import Chunk, Dataset, chunk_document, load_dataset
from .models import RiskModels

_ICR_RATING = re.compile(r"Overall Outcome \| ([A-Za-z ]+?) \|")


def icr_ratings_from_docs(docs) -> dict:
    out = {}
    for d in docs:
        if d["doc_type"] == "ICR":
            m = _ICR_RATING.search(d["text"])
            name = m.group(1) if m and m.group(1) in RATING_NUM else parse_icr_rating(d["text"])   # real ICR text fallback
            if name in RATING_NUM:
                out[d["project_id"]] = RATING_NUM[name]
    return out


class Platform:
    def __init__(self, ds: Dataset):
        from . import doc_links
        self.ds = ds
        doc_links.register(ds.docs)          # PDF links travel with the documents (bronze_docs.meta_json), not a package file
        self.index = HybridIndex()
        self.graph = KnowledgeGraph()
        self.models = RiskModels()
        self.chunks = []
        self.icr = {}
        self.versions = []      # model registry (champion history)

    @classmethod
    def build(cls, data_dir: str, verbose=False):
        ds = load_dataset(data_dir)
        p = cls(ds)
        p.chunks = [c for d in ds.docs for c in chunk_document(d)]
        p.index.build(p.chunks)
        p.graph.build(ds, p.chunks, None)
        p.icr = icr_ratings_from_docs(ds.docs)
        p.models.fit(ds.contracts, open_tenders=ds.open_tenders)
        p.versions.append(dict(version=1, metrics=p.models.metrics, n_train=len(ds.contracts)))
        return p

    @classmethod
    def from_lake(cls, lake, store=None, retriever_factory=None, model_name="procintel_risk"):
        """Rebuild a serving platform from silver/gold tables. retriever_factory(chunks) -> object with .search/.chunks/.boost
        (e.g. VectorSearchRetriever); default is the in-memory HybridIndex. Models come from the registry champion if present."""
        import json
        from dataclasses import fields
        docs = [dict(doc_id=r.doc_id, doc_type=r.doc_type, project_id=r.project_id, filename=r.filename, text=r.text,
                     meta=json.loads(r.meta_json or "{}"), ingested=getattr(r, "_ingest_ts", None))
                for r in lake.read("bronze_docs").drop_duplicates("doc_id", keep="last").itertuples()]
        from .ingest import _dedupe_docs
        docs = _dedupe_docs(docs)                              # identical copies of one report, as in build_dataset
        ds = Dataset(lake.read("silver_projects"), lake.read("silver_contracts"), lake.read("silver_governance"),
                     lake.read("silver_disbursements"), lake.read("silver_isr"), docs, {},
                     lake.read("silver_open_tenders") if lake.exists("silver_open_tenders") else pd.DataFrame())
        if lake.exists("silver_procurement_plan"):            # plan activities for the 'behind its own plan' warning
            from .ingest import _plan_table
            ds.plan = _plan_table(lake.read("silver_procurement_plan"))
        p = cls(ds)
        cols = [f.name for f in fields(Chunk)]
        p.chunks = [Chunk(**r) for r in lake.read("gold_chunks")[cols].to_dict("records")]
        if retriever_factory:
            p.index = retriever_factory(p.chunks)
        else:
            p.index.build(p.chunks)
        p.graph.build(ds, p.chunks, None)
        if lake.exists("gold_feedback_priors"):                # user feedback aggregated by the pipeline (events.py)
            fp = lake.read("gold_feedback_priors")
            p.index.boost.update(dict(zip(fp.chunk_id, fp.boost.astype(float))))
        p.icr = icr_ratings_from_docs(docs)
        if store is not None and store.has(model_name):
            p.models = store.load(model_name)
        else:
            p.models.fit(lake.read("gold_contract_features"), open_tenders=ds.open_tenders)
            if store is not None:
                from .mlops import register_champion
                register_champion(store, model_name, p.models)
        p.versions.append(dict(version=len(p.versions) + 1, metrics=p.models.metrics, n_train=len(ds.contracts)))
        return p

    @property
    def chunks_by_id(self):
        return {c.chunk_id: c for c in self.index.chunks}

    @property
    def country_names(self):
        return self.ds.governance.drop_duplicates("country_code").set_index("country_code").country.to_dict()

    def gov_asof(self, code, date):
        g = self.ds.governance
        gc = g[g.country_code == code].sort_values("available_from")
        g = gc[gc.available_from <= pd.Timestamp(date)]
        if len(g):
            return g.iloc[-1]
        # nothing published yet -> earliest row; no governance data at all (e.g. regional '3E') -> NaNs, models handle missing
        return gc.iloc[0] if len(gc) else pd.Series(np.nan, index=g.columns).astype(object)

    def bias(self):
        return optimism_bias(self.ds.isr, self.icr)
