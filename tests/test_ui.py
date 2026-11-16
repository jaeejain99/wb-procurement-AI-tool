"""FastAPI backend behind the React UI, offline (PROCINTEL_OFFLINE: no model calls), on synthetic data."""
import json, os, urllib.parse, warnings; warnings.filterwarnings("ignore")
import pytest
from fastapi.testclient import TestClient
from procintel.synth import generate, write_dataset


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("ui"); write_dataset(generate(seed=3, n_projects=60), str(d))
    os.environ["PROCINTEL_OFFLINE"] = "1"; os.environ["PROCINTEL_EVENTS_DIR"] = str(d / "events")
    import app
    app.STATE.clear(); app.STATE.update(status="loading", error=None)
    c = TestClient(app.api)
    assert c.get("/api/health").json()["status"] == "loading" and c.get("/api/meta").status_code == 503   # gated while loading
    app.build_state(str(d))
    assert c.get("/api/health").json()["status"] == "ready"
    return c


@pytest.fixture(scope="module")
def pkg(client):
    o = client.get("/api/meta").json()["options"]
    return dict(description="Rehab of rural roads", category="Civil Works", sector=o["sectors"][0], country_code=o["countries"][0]["code"],
                value_usd_m=12, method=o["methods"][0])


def test_meta_and_home_summary(client):
    m = client.get("/api/meta").json()
    assert m["summary"]["passages"] > 100 and "Civil Works" in m["options"]["categories"] and m["offline"]


def test_package_workspace_schedule_market_risks(client, pkg):
    r = client.post("/api/package/analyze", json=pkg); j = r.json()
    assert r.status_code == 200 and j["schedule"]["plan_days"] > 0 and j["market"]["methods"]
    assert all(not m["thin"] for m in j["market"]["methods"] if m.get("fastest") or m.get("lowest_rebid"))   # no badge on thin evidence
    assert "lots" not in j["market"]         # removed until the model can separate lot sizes
    assert all(f["mitigation_source"] == "standard practice" for f in j["findings"] if f["kind"] == "risk")    # offline: no model rewrite
    assert any(f["kind"] == "risk" and f["evidence"][0]["quote"] for f in j["findings"])


@pytest.mark.parametrize("bad,code", [(dict(category="Nope"), 400), (dict(value_usd_m=-5), 422), (dict(country_code="XXX"), 400),
                                      (dict(notice_date="not-a-date"), 400)])
def test_package_input_validation(client, pkg, bad, code):
    r = client.post("/api/package/analyze", json={**pkg, **bad})
    assert r.status_code == code and r.json()["detail"]


def test_portfolio_and_project_page(client):
    j = client.get("/api/portfolio").json()
    assert j["total_active"] > 0 and j["projects"] == sorted(j["projects"], key=lambda p: (-p["risk_score"], -p["overdue_tenders"], p["project_id"]))
    pid = j["projects"][0]["project_id"]
    p = client.get(f"/api/project/{pid}").json()
    assert p["project_id"] == pid and "isr" in p and "open_tenders" in p and "lessons" in p
    assert client.get("/api/project/P000000").status_code == 404


def test_ask_conversation_and_refusal(client):
    r = client.post("/api/ask", json=dict(question="Which contractors in Kenya should we blacklist?")).json()
    assert r["refused"] and r["turn"] == 1 and r["conversation_id"] and "carry" not in r and "draft" not in r
    r2 = client.post("/api/ask", json=dict(question="And who will win the next tender?", conversation_id=r["conversation_id"])).json()
    assert r2["conversation_id"] == r["conversation_id"] and r2["turn"] == 2 and r2["refused"]
    assert client.post("/api/ask", json=dict(question="hi")).status_code == 422


def test_every_portfolio_flag_has_a_plain_language_alert(client):
    p = client.get("/api/portfolio").json()
    for r in p["projects"]:
        assert len(r["alerts"]) == len(r["flags"])
        for a in r["alerts"]:
            assert a["title"] and a["detail"] and a["tech"] in r["flags"]
            assert "P90" not in a["title"] + a["detail"] and "notch" not in a["detail"] and not __import__("re").search(r"\d+d\b", a["detail"])


def test_library_country_is_a_filter_not_a_preference(client):
    countries = [c["name"] for c in client.get("/api/meta").json()["options"]["countries"]]
    for name in countries:
        s = client.get("/api/search", params=dict(q="delays in procurement and contract award", country=name)).json()
        assert all(x["country"] == name for x in s["results"]), name
    assert any(client.get("/api/search", params=dict(q="delays in procurement and contract award", country=n)).json()["results"] for n in countries)


def test_library_threshold_returns_nothing_for_off_topic_queries(client):
    for q in ["chocolate cake recipe", "xyzzy blorf qwerty", "the and of"]:
        assert client.get("/api/search", params=dict(q=q)).json()["results"] == [], q
    s = client.get("/api/search", params=dict(q="benchmark cost estimates before the tender")).json()
    assert s["results"] and all(x["match"] >= s["threshold"] for x in s["results"])


def test_hyphenated_prefix_words_are_one_term_in_a_query():
    from procintel.services import key_terms, snippet
    assert key_terms("re-tendering", query=True) == {"retender"}                     # not just 'tender'
    assert {"retender", "tender"} <= key_terms("the contract was re-tendered twice")   # a passage still matches both
    assert "tender" in key_terms("re-tendering") and "retender" not in key_terms("competitive tendering")
    assert key_terms("cost-benefit analysis", query=True) == {"cost", "benefit", "analysi"}
    long = "Background text. " * 80 + "The works were re-tendered after bids came in high."
    assert "re-tendered" in snippet(long, {"retender"}) and snippet(long, {"retender"}).startswith("…")


def test_relevance_weights_rare_terms_and_stems_plurals():
    from procintel.services import key_terms, match_score, stem
    assert stem("delays") == stem("delayed") == "delay" and stem("bids") == "bid" and stem("rating") == "rating"
    df, n = {"world": 900, "result": 800, "football": 0, "cup": 1}, 1000
    assert match_score(key_terms("football world cup results"), key_terms("World Bank results framework"), df, n) < 0.5


def test_document_view_groups_the_report_by_page_and_contains_the_passage(client):
    hit = client.get("/api/search", params=dict(q="benchmark cost estimates before the tender")).json()["results"][0]
    d = client.get("/api/document/" + hit["doc_id"]).json()
    pages = [pg["page"] for pg in d["pages"]]
    assert pages == sorted(pages) and d["passages"] == sum(len(pg["passages"]) for pg in d["pages"])
    assert any(p["chunk_id"] == hit["chunk_id"] for pg in d["pages"] for p in pg["passages"])
    assert client.get("/api/document/NOPE_P000000_1").status_code == 404


def test_pdf_links_open_the_passage_page():
    from procintel import doc_links
    doc_links._CACHE = {"ICR_P160162_34360386": "https://documents.worldbank.org/x.pdf"}
    try:
        assert doc_links.page_link("ICR_P160162_34360386", 28) == "https://documents.worldbank.org/x.pdf#page=28"
        assert doc_links.page_link("ICR_P000000_1", 3) is None and doc_links.wds_id("ICR_P160162_34360386") == "34360386"
    finally:
        doc_links._CACHE = None


def test_library_search_passage_feedback_and_insights(client):
    s = client.get("/api/search", params=dict(q="benchmark cost estimates before the tender", section="lessons")).json()
    assert s["results"] and all(x["section_kind"] == "lessons" for x in s["results"])
    cid = s["results"][0]["chunk_id"]
    p = client.get("/api/passage/" + urllib.parse.quote(cid, safe="")).json()
    assert p["chunk_id"] == cid and p["text"]
    assert client.post("/api/feedback", json=dict(chunk_ids=[cid], useful=True, surface="library")).json()["boosts"][cid] > 0
    i = client.get("/api/insights").json()
    assert i["model"]["version"] == 4 and i["data"]["passages"] > 100 and i["usage"]["feedback_total"] >= 1 and i["whats_new"]
    from procintel.events import feedback_priors, load_events
    import app
    ev = load_events(app.STATE["events"].local_dir)
    assert {"search", "feedback", "analyze", "ask"} <= {e["event"] for e in ev} and all("@" not in e["user"] for e in ev)
    assert feedback_priors(ev)[cid] > 0


def test_p95_is_never_below_the_median():
    from procintel.events import usage_kpis
    k = usage_kpis([dict(event="ask", user="u", day="2026-11-03", ms=m) for m in (10600, 11800)])
    assert k["p95_answer_ms"] >= k["median_answer_ms"]


def test_ui_is_served_when_built(client):
    import app
    if not os.path.exists(os.path.join(app.DIST, "index.html")):
        pytest.skip("web/dist not built")
    r = client.get("/")
    assert r.status_code == 200 and '<div id="root">' in r.text
    assert client.get("/package").status_code == 200 and client.get("/api/nope").status_code == 404


def _sse(resp):
    return [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]


def test_ask_stream_sends_progress_and_only_verified_text(client, monkeypatch):
    """Stream a full agent turn with a scripted model: progress events, verified sentences only, and a done event whose
    answer is exactly the sentences that were streamed."""
    import app, re
    class Model:
        def chat(self, model, messages, tools=None, **kw):
            u = [m["content"] for m in messages if m["role"] == "user"][-1]
            ids = re.findall(r"^\[([^\]]+#\d+)\]", u.split("EVIDENCE", 1)[1], re.M)
            line = next(l for l in u.splitlines() if l.startswith(f"[{ids[0]}]"))
            first = line.split(" | ", 1)[1].split(".")[0]
            return {"content": f"{first}. [{ids[0]}]\nIt took 4242 days [{ids[0]}]."}
    class Judge:
        def __call__(self, items): return [(True, "ok")] * len(items)
    agent = app.STATE["agent"]
    monkeypatch.setattr(agent, "client", Model()); monkeypatch.setattr(agent, "judge", Judge())
    r = client.post("/api/ask/stream", json=dict(question="What do completion reports say about cost estimates for civil works?"))
    ev = _sse(r)
    kinds = [e["type"] for e in ev]
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert kinds[0] == "status" and "tool" in kinds and kinds[-1] == "done"
    streamed = [e["text"] for e in ev if e["type"] == "sentence" and not e["meta"]]
    done = ev[-1]["result"]
    assert streamed == [a["text"] for a in done["answer"] if not a.get("meta")] and streamed
    assert not any("4242" in t for t in streamed) and done["removed_count"] >= 1 and done["removed"] == []   # removed, never shown
    assert "4242" not in r.text                               # nor sent: removed sentences reach the browser only as a count
    assert {"draft", "carry", "plan"}.isdisjoint(done) and done["tools_run"]


def test_ask_stream_refusal(client):
    ev = _sse(client.post("/api/ask/stream", json=dict(question="Which contractors in Kenya should we blacklist?")))
    assert ev[-1]["type"] == "done" and ev[-1]["result"]["refused"] and not any(e["type"] == "sentence" for e in ev)


def test_reload_keeps_serving_on_failure_and_keeps_conversations(client, tmp_path):
    import app
    plat, convs = app.STATE["plat"], app.STATE["convs"]
    convs["c1"] = ["kept"]
    app.build_state(str(tmp_path / "missing"), reload=True)              # a broken snapshot: the current data stays
    assert app.STATE["status"] == "ready" and app.STATE["plat"] is plat and app.STATE["reload_error"]
    assert client.get("/api/health").json()["reload_error"]
    convs.pop("c1")
