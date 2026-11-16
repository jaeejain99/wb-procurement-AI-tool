"""Databricks Mosaic AI Vector Search as a retrieval backend.

The delta-sync index is built on `gold_chunks_vs` (the decision-relevant subset, Change Data Feed on) and embeds `embed_text`
with `databricks-gte-large-en`; it re-syncs itself as the pipeline merges and deletes chunks.
  ensure_vector_index / sync_and_wait  - create and sync the index (notebooks 04, 06; verified on the workspace)
  VectorSearchRetriever                - HybridIndex-compatible .search over an index handle (SDK client or RestVectorIndex)
  RestVectorIndex                      - the same query over the REST API, with the auth the app already has (no extra package)
  FusedRetriever                       - what the app and agent use: Vector Search (semantic, subset) and the in-memory hybrid
                                         index (lexical, full corpus) fused by reciprocal rank; falls back to in-memory alone
                                         if Vector Search fails, and says which backend answered."""
from __future__ import annotations
from .index import soft_boost


def ensure_vector_index(index_name, endpoint, source_table, embed_endpoint="databricks-gte-large-en", client=None):
    """Create endpoint + delta-sync index if missing (idempotent), return the index handle."""
    if client is None:
        from databricks.vector_search.client import VectorSearchClient
        client = VectorSearchClient()
    try:
        client.get_endpoint(endpoint)
    except Exception:
        client.create_endpoint(name=endpoint, endpoint_type="STANDARD")
    try:
        return client.get_index(endpoint_name=endpoint, index_name=index_name)
    except Exception:
        return client.create_delta_sync_index(endpoint_name=endpoint, index_name=index_name, source_table_name=source_table,
                                              pipeline_type="TRIGGERED", primary_key="chunk_id",
                                              embedding_source_column="embed_text", embedding_model_endpoint_name=embed_endpoint)


def sync_and_wait(index, timeout_s=7200, poll_s=30, raise_on_timeout=True):
    """Block until the index is queryable and has picked up the latest gold_chunks. A brand-new delta-sync index embeds the
    table by itself (slow on first build) and rejects sync requests until then ('not ready'), so: wait while it is building,
    then trigger a sync (retrying while not ready), then wait for that sync to finish. raise_on_timeout=False returns the
    last status instead (the index keeps building server-side; callers fall back to the in-memory index meanwhile)."""
    import time
    t0 = time.time()

    def status():
        st = index.describe().get("status", {})
        busy = any(k in str(st.get("detailed_state", "")) for k in ("PROVISIONING", "INITIALIZING", "TRIGGERED_UPDATE", "UPDATING"))
        return st, bool(st.get("ready")) and not busy

    def wait_idle():
        while time.time() - t0 < timeout_s:
            st, idle = status()
            if idle:
                return st
            print(f"[vector index] {st.get('detailed_state', '')} rows={st.get('indexed_row_count')}"); time.sleep(poll_s)
        if raise_on_timeout:
            raise TimeoutError(f"vector index not ready after {timeout_s}s: {status()[0]}")
        return None

    if wait_idle() is None:
        return status()[0]
    while True:
        try:
            index.sync(); break
        except Exception as e:
            if "not ready" not in str(e):
                raise
            if time.time() - t0 > timeout_s:
                if raise_on_timeout:
                    raise
                return status()[0]
            time.sleep(poll_s)
    time.sleep(poll_s)                                     # let the triggered update register before polling
    return wait_idle() or status()[0]


class VectorSearchRetriever:
    """Same interface the Copilot uses on HybridIndex: .search(query, k, filters, prefer) -> [(chunk, score, comps)], .chunks, .boost."""
    MAX_RESULTS = 200          # hybrid queries reject num_results > 200 with HTTP 400 (verified on the workspace, 2026-11-04)

    def __init__(self, index, chunks: list, over_fetch=5, hybrid=True):
        self.index, self.chunks, self.over_fetch, self.hybrid = index, chunks, over_fetch, hybrid
        self.by_id = {c.chunk_id: c for c in chunks}
        self.boost: dict = {}

    def add(self, new_chunks):                    # the index itself syncs from Delta CDF; only the local text cache is refreshed
        for c in new_chunks:
            if c.chunk_id not in self.by_id:
                self.chunks.append(c)
            self.by_id[c.chunk_id] = c
        self.index.sync() if hasattr(self.index, "sync") else None
        return dict(added=len(new_chunks), refit=False, oov=0.0)

    def search(self, query, k=8, filters=None, prefer=None, now_year=2026):
        filters, prefer = filters or {}, prefer or {}
        flt = {col: sorted(vals) for col, vals in filters.items()}                  # list value == IN (verified on the workspace)
        res = self.index.similarity_search(query_text=query, columns=["chunk_id"], filters=flt or None,
                                          num_results=min(k * self.over_fetch, self.MAX_RESULTS),
                                          query_type="HYBRID" if self.hybrid else "ANN")
        rows = (res.get("result") or {}).get("data_array") or []                    # [[chunk_id, score], ...]; absent when empty
        if not rows:
            return []
        top = max(r[-1] for r in rows) or 1.0
        out = []
        for r in rows:
            c = self.by_id.get(r[0])
            if c is None:
                continue
            s = float(r[-1])
            # NOTE: scores are RRF-scale on hybrid queries, so they are normalised to the top hit; the stem-overlap gate in the
            # Copilot is the real relevance check on this backend.
            out.append((c, s / top * soft_boost(c, prefer, now_year, self.boost) * 100, dict(sparse=s / top, dense=s / top)))
        out.sort(key=lambda t: -t[1])
        return out[:k]


def index_ready(index) -> bool:
    try:
        return bool(index.describe().get("status", {}).get("ready"))
    except Exception:
        return False


class RestVectorIndex:
    """similarity_search over POST /api/2.0/vector-search/indexes/{name}/query, same result shape as the SDK."""

    def __init__(self, index_name, client=None, timeout=30):
        from .llm import shared_client
        self.name, self.client, self.timeout = index_name, client or shared_client(), timeout

    def _call(self, method, path, body=None):
        import json, urllib.request
        host, headers = self.client._auth()
        req = urllib.request.Request(f"{host}{path}", method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={**headers, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self.client.ctx) as r:
            return json.loads(r.read())

    def describe(self):
        return self._call("GET", f"/api/2.0/vector-search/indexes/{self.name}")

    def similarity_search(self, query_text, columns, filters=None, num_results=10, query_type="HYBRID"):
        import json
        body = dict(query_text=query_text, columns=columns, num_results=num_results, query_type=query_type)
        if filters:
            body["filters_json"] = json.dumps(filters)
        return self._call("POST", f"/api/2.0/vector-search/indexes/{self.name}/query", body)


class FusedRetriever:
    """Vector Search + in-memory hybrid index, fused by reciprocal rank. Same interface as HybridIndex."""

    def __init__(self, memory_index, vector_retriever=None):
        self.mem, self.vs = memory_index, vector_retriever
        self.chunks, self.boost = memory_index.chunks, memory_index.boost      # one feedback prior for both backends
        if self.vs is not None:
            self.vs.boost = self.boost
        self.last_backend, self.vs_errors = "in-memory", 0

    @property
    def ids(self):
        return self.mem.ids

    def add(self, new_chunks):
        return self.mem.add(new_chunks)

    def search(self, query, k=8, filters=None, prefer=None, now_year=2026):
        from concurrent.futures import ThreadPoolExecutor
        vs, fut = [], None
        pool = ThreadPoolExecutor(max_workers=1) if self.vs is not None else None
        if pool:                                       # both backends at once: Vector Search over the network, memory locally
            fut = pool.submit(self.vs.search, query, k * 2, filters, prefer, now_year)
        mem = self.mem.search(query, k=k * 2, filters=filters, prefer=prefer, now_year=now_year)
        if fut is not None:
            try:
                vs = fut.result(timeout=20)
            except Exception:                          # endpoint down, permission, timeout: keep answering from memory
                self.vs_errors += 1
            pool.shutdown(wait=False)
        if not vs:
            self.last_backend = "in-memory"
            return mem[:k]
        self.last_backend = "vector search + in-memory"
        fused, info = {}, {}
        for lst, tag in ((vs, "vs"), (mem, "mem")):
            for r, (c, s, comp) in enumerate(lst):
                fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (60 + r)
                prev = info.get(c.chunk_id, (c, {}))[1]
                info[c.chunk_id] = (c, {**prev, **comp, f"in_{tag}": True})
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(info[i][0], fused[i] * 100, info[i][1]) for i in top]
