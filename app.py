"""ProcIntel web app: FastAPI backend + React UI (web/, built to web/dist).

    python app.py [--data DIR | --snapshot DIR] [--port 8000]
      no --data       synthetic demo data (generated into data/ui on first run)
      --data DIR      a folder in the procintel layout (e.g. the output of procintel.live_wb)
      --snapshot DIR  a lakehouse snapshot exported by Databricks notebook 05 (tables + UC champion model)
As a Databricks App: PROCINTEL_SNAPSHOT_VOLUME is downloaded with the app's service principal, PROCINTEL_VS_INDEX fuses Mosaic AI
Vector Search into retrieval, events go to PROCINTEL_EVENTS_VOLUME, and the server listens on DATABRICKS_APP_PORT.

Robustness: the platform loads in a background thread (GET /api/health reports loading / ready / error and the UI waits on it),
requests are validated with pydantic, every read runs concurrently (only conversation and feedback state is locked), errors come
back as JSON with a message, and every action is logged as a usage event without ever failing the request."""
from __future__ import annotations
import argparse, json, os, re, shutil, threading, time, traceback, uuid, warnings
from collections import OrderedDict
from typing import Literal, Optional

warnings.filterwarnings("ignore")
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from procintel import services
from procintel.events import EventLog, feedback_priors, load_events, usage_kpis

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "web", "dist")
STATE: dict = {"status": "loading", "error": None}
CONV_LOCK, FEEDBACK_LOCK = threading.Lock(), threading.Lock()
MAX_CONVERSATIONS, MEMORY_TURNS = 200, 4


# ------------------------------------------------------------------ platform loading
def fetch_volume(volume_dir: str, dest: str) -> str:
    """Copy a UC Volume folder to local disk through the Files API (Apps have no /Volumes mount)."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    def walk(src, rel):
        for e in w.files.list_directory_contents(src):
            if e.is_directory:
                walk(e.path, os.path.join(rel, e.name))
            else:
                os.makedirs(os.path.join(dest, rel), exist_ok=True)
                with open(os.path.join(dest, rel, e.name), "wb") as f:
                    f.write(w.files.download(e.path).contents.read())
    walk(volume_dir.rstrip("/"), "")
    return dest


def attach_vector_search(plat) -> str:
    name = os.getenv("PROCINTEL_VS_INDEX")
    if not name:
        return "in-memory hybrid (TF-IDF + LSA)"
    from procintel.vectorstore import FusedRetriever, RestVectorIndex, VectorSearchRetriever
    plat.index = FusedRetriever(plat.index, VectorSearchRetriever(RestVectorIndex(name), plat.index.chunks))
    return f"Mosaic AI Vector Search ({name}) fused with the in-memory hybrid index"


def build_state(data_dir: str | None = None, snapshot: str | None = None, reload=False):
    """Load data and models and wire the copilot, agent, judge and event log. Sets STATE['status'] to ready or error.
    reload=True (a newer snapshot): the app keeps serving the current data while the new one loads, swaps it in in one
    update, keeps open conversations, and on failure keeps the current data (the error goes to STATE['reload_error'])."""
    try:
        from procintel import Copilot, Platform
        from procintel.agent import Agent
        from procintel.guardrails import LLMJudge
        from procintel.learning import ContinuousLearner
        from procintel.llm import OfflineChat
        offline = bool(os.getenv("PROCINTEL_OFFLINE"))
        if snapshot:
            from procintel.lakehouse import LocalLake
            from procintel.mlops import LocalModelStore
            plat = Platform.from_lake(LocalLake(snapshot), LocalModelStore(os.path.join(snapshot, "models")))
            source = "Databricks lakehouse snapshot"
        else:
            if data_dir is None:
                data_dir = os.path.join(HERE, "data", "ui")
                if not os.path.exists(os.path.join(data_dir, "contracts.csv")):
                    from procintel.synth import generate, write_dataset
                    write_dataset(generate(seed=7, n_projects=110), data_dir)
                source = "synthetic demo data"
            else:
                source = f"data folder {os.path.basename(os.path.normpath(data_dir))}"
            plat = Platform.build(data_dir)
        retrieval = attach_vector_search(plat)
        local_events = os.getenv("PROCINTEL_EVENTS_DIR") or os.path.join(HERE, "data", "events")
        events = EventLog(local_events, os.getenv("PROCINTEL_EVENTS_VOLUME"))
        plat.index.boost.update(feedback_priors(load_events(local_events)))
        judge = None if offline else LLMJudge()
        if not offline:                                          # warm up auth once, before parallel tool calls need it
            try:
                from procintel.llm import shared_client
                shared_client()._auth()
            except Exception:
                pass
        STATE.update(plat=plat, cop=Copilot(plat, judge=judge), learner=ContinuousLearner(plat), agent=Agent(plat, client=OfflineChat() if offline else None, use_judge=not offline),
                     events=events, retrieval=retrieval, source=source, offline=offline, status="ready", reload_error=None,
                     convs=STATE["convs"] if reload and STATE.get("convs") is not None else OrderedDict(),
                     loaded_at=time.strftime("%Y-%m-%d %H:%M"))
        # library relevance threshold needs corpus term statistics (~4 s on 29k passages): warm them now, not on the first search
        threading.Thread(target=lambda: services._term_stats(STATE), daemon=True).start()
    except Exception as e:                                       # keep serving: /api/health shows what went wrong
        if reload:
            STATE.update(reload_error=f"{type(e).__name__}: {e}"); return
        STATE.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])


SNAPSHOT_VERSION = "_version.txt"           # written last by serving.export_snapshot


def snapshot_version(volume_dir: str) -> str | None:
    from databricks.sdk import WorkspaceClient
    try:
        return WorkspaceClient().files.download(f"{volume_dir.rstrip('/')}/{SNAPSHOT_VERSION}").contents.read().decode().strip()
    except Exception:
        return None                                              # a snapshot exported before versioning, or not reachable


def watch_snapshot(volume_dir: str, every_s: int):
    """Poll the snapshot's version; when a pipeline or refresh run exports a newer one, load it without a restart."""
    while True:
        time.sleep(every_s)
        v = snapshot_version(volume_dir)
        if not v or v == STATE.get("snapshot_version") or STATE["status"] != "ready":
            continue
        try:
            snap = fetch_volume(volume_dir, os.path.join(HERE, "data", f"snapshot_{re.sub(r'[^0-9A-Za-z]', '', v)}"))
        except Exception as e:
            STATE.update(reload_error=f"snapshot download failed: {e}"); continue
        build_state(None, snap, reload=True)
        if not STATE.get("reload_error"):
            STATE["snapshot_version"] = v
        keep = snap if not STATE.get("reload_error") else STATE.get("snapshot_dir")
        STATE["snapshot_dir"] = keep
        for d in os.listdir(os.path.join(HERE, "data")):            # data is in memory: older copies only take disk space
            full = os.path.join(HERE, "data", d)
            if (d == "snapshot" or d.startswith("snapshot_")) and full != keep:
                shutil.rmtree(full, ignore_errors=True)


def ready():
    if STATE["status"] != "ready":
        raise HTTPException(503, detail="Tenderwise is still loading data and models" if STATE["status"] == "loading"
                            else f"Tenderwise failed to load: {STATE['error']}")
    return STATE


def log_event(request: Request, event: str, t0: float, **fields):
    try:
        STATE["events"].log(event, request.headers.get("X-Forwarded-Email"), ms=round(1000 * (time.time() - t0)), **fields)
    except Exception:
        pass


# ------------------------------------------------------------------ request models
class PackageIn(BaseModel):
    description: str = Field("", max_length=300)
    category: str
    sector: str
    country_code: str
    value_usd_m: Optional[float] = Field(None, gt=0, lt=100000)
    method: str
    review_type: Optional[str] = "Prior"
    notice_date: Optional[str] = None
    es_risk: Optional[str] = "Moderate"


class AskIn(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    conversation_id: Optional[str] = Field(None, max_length=64)
    context: Optional[dict] = None              # e.g. the package from the workspace ("Ask about this package")


class FeedbackIn(BaseModel):
    chunk_ids: list[str] = Field(..., max_length=20)
    useful: bool
    surface: Literal["package", "agent", "library", "tender"] = "agent"
    question: Optional[str] = Field("", max_length=300)


# ------------------------------------------------------------------ API
api = FastAPI(title="Tenderwise API", version="2.0", docs_url="/api/docs", openapi_url="/api/openapi.json")


@api.exception_handler(ValueError)
async def _value_error(_, e: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(e)})


@api.exception_handler(KeyError)
async def _key_error(_, e: KeyError):
    return JSONResponse(status_code=404, content={"detail": str(e).strip("'\"")})


@api.get("/api/health")
def health():
    return dict(status=STATE["status"], error=STATE.get("error"), source=STATE.get("source"), loaded_at=STATE.get("loaded_at"),
                snapshot_version=STATE.get("snapshot_version"), reload_error=STATE.get("reload_error"))


@api.get("/api/meta")
def meta():
    s = ready(); p = s["plat"]
    return dict(source=s["source"], retrieval=s["retrieval"], offline=s["offline"], options=services.options(p),
                summary=dict(contracts=len(p.ds.contracts), projects=len(p.ds.projects), active=int((p.ds.projects.status == "Active").sum()),
                             documents=len(p.ds.docs), passages=len(p.index.chunks), countries=len(set(p.ds.contracts.country_code)),
                             open_tenders=len(getattr(p.ds, "open_tenders", []))))


@api.post("/api/package/analyze")
def package_analyze(body: PackageIn, request: Request):
    s, t0 = ready(), time.time()
    rep = services.analyze_package(s, body.model_dump())
    log_event(request, "analyze", t0, category=body.category, country=body.country_code,
              risks=sum(f.get("kind") == "risk" for f in rep["findings"]))
    return rep


@api.get("/api/portfolio")
def portfolio(request: Request, country: Optional[str] = None, as_of: Optional[str] = None, top: int = Query(50, ge=1, le=200)):
    s, t0 = ready(), time.time()
    r = services.portfolio(s, country or None, as_of or None, top)
    log_event(request, "portfolio", t0, country=country)
    return r


@api.get("/api/project/{pid}")
def project(pid: str, request: Request):
    s, t0 = ready(), time.time()
    r = services.project_page(s, pid)
    log_event(request, "project", t0, project_id=pid)
    return r


def _conversation(s, body: AskIn):
    cid = body.conversation_id or uuid.uuid4().hex
    with CONV_LOCK:
        history = list(s["convs"].get(cid, []))
    context = body.context if (body.context and not history) else None     # package from the workspace: first turn only
    return cid, history, context


def _finish(s, request, body, cid, r, t0):
    """Remember the verified turn (and its plan, for follow-ups), strip server-only fields, log the event."""
    if r.get("mode") in ("agent", "policy"):
        turn = dict(question=r["question"], answer=r["answer"], refused=r.get("refused"), carry=r.get("carry", {}), plan=r.get("plan"))
        with CONV_LOCK:
            s["convs"][cid] = (s["convs"].get(cid, []) + [turn])[-MEMORY_TURNS:]
            s["convs"].move_to_end(cid)
            while len(s["convs"]) > MAX_CONVERSATIONS:
                s["convs"].popitem(last=False)
    with CONV_LOCK:
        turns = len(s["convs"].get(cid, []))
    for k in ("draft", "carry", "plan"):                     # the unverified draft and full source texts never leave the server
        r.pop(k, None)
    log_event(request, "ask", t0, question=body.question[:300], conversation_id=cid, turn=turns, mode=r.get("mode"),
              refused=r.get("refused"), held_back=r.get("held_back"), kept=sum(not a.get("meta") for a in r.get("answer", [])),
              removed=len(r.get("removed", [])), removed_text=[x.get("text", "")[:300] for x in r.get("removed", [])],
              tools_run=r.get("tools_run"))
    # sentences the verifier removed are unverified text: logged above for review, only their count goes to the browser
    r = dict(r, removed_count=len(r.get("removed", [])), removed=[])
    return dict(r, conversation_id=cid, turn=turns)


@api.post("/api/ask")
def ask(body: AskIn, request: Request):
    """Non-streaming answer (same pipeline as /api/ask/stream)."""
    s, t0 = ready(), time.time()
    cid, history, context = _conversation(s, body)
    r = s["agent"].ask(body.question.strip(), history=history, context=context)
    return _finish(s, request, body, cid, r, t0)


@api.post("/api/ask/stream")
def ask_stream(body: AskIn, request: Request):
    """Server-sent events: status, tool, sentence (VERIFIED sentences only), removed (a count, no text), done. Unverified text never
    leaves the server."""
    s, t0 = ready(), time.time()
    cid, history, context = _conversation(s, body)

    def events():
        try:
            for ev in s["agent"].stream(body.question.strip(), history=history, context=context):
                if ev["type"] == "done":
                    ev = {"type": "done", "result": _finish(s, request, body, cid, ev["result"], t0)}
                elif ev["type"] == "removed":                 # not its text: only verified sentences are sent
                    ev = {"type": "removed"}
                yield f"data: {json.dumps(ev, default=str, ensure_ascii=False)}\n\n"
        except Exception as e:                                # tell the browser, so it can fall back to /api/ask
            yield f"data: {json.dumps({'type': 'error', 'detail': f'{type(e).__name__}: {e}'})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api.get("/api/search")
def search(request: Request, q: str = Query(..., min_length=3, max_length=300), k: int = Query(10, ge=1, le=30),
           doc_type: str = "", section: str = "", country: str = ""):
    s, t0 = ready(), time.time()
    r = services.search(s, q, k, doc_type, section, country)
    log_event(request, "search", t0, query=q[:200], hits=len(r["results"]))
    return r


@api.get("/api/passage/{chunk_id:path}")
def passage(chunk_id: str):
    return services.passage(ready(), chunk_id)


@api.get("/api/document/{doc_id}")
def document(doc_id: str):
    return services.document(ready(), doc_id)


@api.post("/api/feedback")
def feedback(body: FeedbackIn, request: Request):
    s, t0 = ready(), time.time()
    with FEEDBACK_LOCK:
        s["learner"].feedback(body.chunk_ids, body.useful)
    log_event(request, "feedback", t0, chunk_ids=body.chunk_ids, useful=body.useful, surface=body.surface, question=body.question)
    return dict(ok=True, boosts={i: s["plat"].index.boost.get(i, 0) for i in body.chunk_ids})


@api.get("/api/insights")
def insights():
    return services.insights(ready())


@api.get("/api/kpis")
def kpis():
    s = ready()
    return usage_kpis(load_events(s["events"].local_dir))


# ------------------------------------------------------------------ React UI (built by `npm run build` in web/)
if os.path.isdir(os.path.join(DIST, "assets")):
    api.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")


@api.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, detail="not found")
    index = os.path.join(DIST, "index.html")
    if not os.path.exists(index):
        return JSONResponse(status_code=503, content={"detail": "UI not built: run `npm install && npm run build` in web/"})
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


def start(data_dir=None, snapshot=None, background=True):
    if background:
        threading.Thread(target=build_state, args=(data_dir, snapshot), daemon=True).start()
    else:
        build_state(data_dir, snapshot)


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--snapshot")
    ap.add_argument("--port", type=int, default=int(os.getenv("DATABRICKS_APP_PORT", 8000)))
    a = ap.parse_args()
    host = "0.0.0.0" if os.getenv("DATABRICKS_APP_PORT") else "127.0.0.1"
    vol = os.getenv("PROCINTEL_SNAPSHOT_VOLUME")

    def boot():
        snap = a.snapshot
        if vol and not snap:
            try:
                ver = snapshot_version(vol)                  # read before the copy: a newer export during it is picked up later
                snap = fetch_volume(vol, os.path.join(HERE, "data", "snapshot"))
            except Exception as e:
                STATE.update(status="error", error=f"snapshot download failed: {e}"); return
            STATE.update(snapshot_version=ver, snapshot_dir=snap)
        build_state(a.data, snap)
        if vol and not a.snapshot:                           # pick up snapshots exported after start-up (refresh / pipeline runs)
            watch_snapshot(vol, int(os.getenv("PROCINTEL_SNAPSHOT_POLL_S", 900)))
    threading.Thread(target=boot, daemon=True).start()
    print(f"Tenderwise listening on http://{host}:{a.port} (data loads in the background; see /api/health)")
    uvicorn.run(api, host=host, port=a.port, log_level="warning")
