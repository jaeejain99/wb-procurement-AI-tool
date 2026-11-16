"""Knowledge graph that reconciles disjoint keys. Typed nodes/edges with link provenance and confidence.

Nodes : Country, Sector, Project, Contract, Supplier, Document, Chunk
Edges : IN_COUNTRY, IN_SECTOR, HAS_CONTRACT, AWARDED_TO, HAS_DOC, HAS_CHUNK,
        EVIDENCE(contract -> chunk, link_type, confidence)   <- the key reconciliation edge
Link types: 'id_mention' (chunk text names the contract id, conf 1.0) > 'same_project_section' (0.8) >
            'sector_country_temporal' (<=0.5, computed lazily at query time to avoid an O(contracts x chunks) edge explosion)."""
from __future__ import annotations
import networkx as nx
import pandas as pd
from .ingest import CID_RE


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def build(self, ds, chunks, country_names):
        self.add_structural(ds.projects, ds.contracts)
        self.add_chunks(chunks, ds.contracts)
        return self

    def add_structural(self, projects, contracts):
        g = self.g
        for _, p in projects.iterrows():
            g.add_node(("Project", p.project_id), status=p.status, country=p.country_code, sector=p.sector,
                       approval=p.approval_date, closing=p.closing_date)
            g.add_edge(("Project", p.project_id), ("Country", p.country_code), type="IN_COUNTRY")
            g.add_edge(("Project", p.project_id), ("Sector", p.sector), type="IN_SECTOR")
        for _, c in contracts.iterrows():
            if ("Contract", c.contract_id) in g:
                continue                                   # idempotent
            g.add_node(("Contract", c.contract_id), category=c.category, award=c.award_date, sector=c.sector, country=c.country_code)
            g.add_edge(("Project", c.project_id), ("Contract", c.contract_id), type="HAS_CONTRACT")
            g.add_edge(("Contract", c.contract_id), ("Supplier", c.supplier_id), type="AWARDED_TO")
            act = getattr(c, "step_activity", None)
            if isinstance(act, str) and act:                   # STEP procurement activity: the key shared with tender notices
                g.add_edge(("Project", c.project_id), ("Activity", act), type="HAS_ACTIVITY")
                g.add_edge(("Activity", act), ("Contract", c.contract_id), type="RESULTED_IN")

    def add_chunks(self, chunks, contracts):
        g = self.g
        by_project = {pid: grp for pid, grp in contracts.groupby("project_id")}
        for ch in chunks:
            d, c = ("Document", ch.doc_id), ("Chunk", ch.chunk_id)
            if c in g:
                continue                                   # idempotent upsert
            g.add_node(d, doc_type=ch.doc_type)
            g.add_edge(("Project", ch.project_id), d, type="HAS_DOC")
            g.add_edge(d, c, type="HAS_CHUNK")
            mentioned = set(CID_RE.findall(ch.text))
            for cid in mentioned:
                if ("Contract", cid) in g:
                    g.add_edge(("Contract", cid), c, type="EVIDENCE", link_type="id_mention", confidence=1.0)
            if ch.section_kind in ("lessons", "procurement") and ch.project_id in by_project:
                for cid in by_project[ch.project_id].contract_id:
                    if cid not in mentioned:
                        g.add_edge(("Contract", cid), c, type="EVIDENCE", link_type="same_project_section", confidence=0.8)

    # -- queries
    def evidence_for_contract(self, cid, min_conf=0.0):
        out = []
        for _, ch, a in self.g.out_edges(("Contract", cid), data=True):
            if a.get("type") == "EVIDENCE" and a["confidence"] >= min_conf:
                out.append((ch[1], a["link_type"], a["confidence"]))
        return sorted(out, key=lambda t: -t[2])

    def project_docs(self, pid, doc_type=None):
        docs = [n[1] for _, n in self.g.out_edges(("Project", pid)) if n[0] == "Document"]
        return [d for d in docs if doc_type is None or self.g.nodes[("Document", d)].get("doc_type") == doc_type]

    def supplier_history(self, supplier_id):
        return [c[1] for c, _ in self.g.in_edges(("Supplier", supplier_id)) if c[0] == "Contract"]

    def temporal_neighbours(self, sector, country, around: pd.Timestamp, years=3):
        """Lazy, low-confidence link: closed projects in the same sector+country whose closing date is within +-years."""
        res = []
        for n, a in self.g.nodes(data=True):
            if n[0] == "Project" and a.get("sector") == sector and a.get("country") == country and a.get("status") == "Closed":
                gap = abs((a["closing"] - around).days) / 365
                if gap <= years:
                    res.append((n[1], round(0.5 * (1 - gap / (years + 1)), 2)))
        return sorted(res, key=lambda t: -t[1])

    def edges_df(self) -> pd.DataFrame:
        """Flat edge list = the gold_kg_edges table (queryable in SQL; loadable into GraphFrames/Neo4j at scale)."""
        rows = [dict(src_type=u[0], src_id=u[1], dst_type=v[0], dst_id=v[1], rel=a["type"], link_type=a.get("link_type"),
                     confidence=a.get("confidence")) for u, v, a in self.g.edges(data=True)]
        return pd.DataFrame(rows)

    def stats(self):
        from collections import Counter
        return dict(nodes=dict(Counter(n[0] for n in self.g.nodes)), edges=dict(Counter(a["type"] for *_, a in self.g.edges(data=True))))
