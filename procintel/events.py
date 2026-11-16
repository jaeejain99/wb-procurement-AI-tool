"""Usage and feedback events: the adoption KPIs and the retrieval feedback loop.

The app writes one small JSON event per action (ask, analyze, search, alerts, conflicts, feedback) to a local folder and, on
Databricks, to a Unity Catalog Volume (one file per event, uploaded off the request thread). Notebook 05 turns the events into
gold_usage_events, gold_feedback_priors and gold_kpis, and ships the priors in the app snapshot, so feedback survives restarts
and reaches every user. Users are stored as a salted hash of the email the Databricks Apps proxy forwards, never the email."""
from __future__ import annotations
import collections, datetime as dt, glob, hashlib, io, json, os, statistics, threading, uuid

STEP, CAP = 0.1, 0.5


def user_key(email: str | None) -> str:
    return hashlib.sha256(("procintel:" + (email or "anonymous").lower()).encode()).hexdigest()[:12]


class EventLog:
    def __init__(self, local_dir: str, volume_dir: str | None = None):
        self.local_dir, self.volume_dir = local_dir, (volume_dir or "").rstrip("/") or None
        os.makedirs(local_dir, exist_ok=True)
        self._w, self.upload_errors = None, 0

    def log(self, event: str, user: str | None = None, **fields) -> dict:
        now = dt.datetime.now(dt.timezone.utc)
        e = dict(ts=now.isoformat(timespec="seconds"), day=now.date().isoformat(), event=event, user=user_key(user), **fields)
        with open(os.path.join(self.local_dir, f"{e['day']}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(e, default=str) + "\n")
        if self.volume_dir:
            threading.Thread(target=self._upload, args=(e,), daemon=True).start()
        return e

    def _upload(self, e):
        try:
            if self._w is None:
                from databricks.sdk import WorkspaceClient
                self._w = WorkspaceClient()
            path = f"{self.volume_dir}/day={e['day']}/{e['ts'].replace(':', '')}-{uuid.uuid4().hex[:8]}.json"
            self._w.files.upload(path, io.BytesIO(json.dumps(e, default=str).encode()), overwrite=True)
        except Exception:
            self.upload_errors += 1


def load_events(folder: str) -> list[dict]:
    """Local .jsonl day files or Volume one-event .json files, whichever the folder holds."""
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "**", "*.json*"), recursive=True)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def feedback_priors(events) -> dict:
    """chunk_id -> retrieval boost: +0.1 per useful, -0.1 per not useful, capped at +-0.5."""
    acc = collections.defaultdict(float)
    for e in events:
        if e.get("event") == "feedback":
            for cid in e.get("chunk_ids") or []:
                acc[cid] += STEP if e.get("useful") else -STEP
    return {k: max(-CAP, min(CAP, v)) for k, v in acc.items()}


def usage_kpis(events, as_of: dt.date | None = None) -> dict:
    as_of = as_of or dt.date.today()
    week = {(as_of - dt.timedelta(days=i)).isoformat() for i in range(7)}
    acts = [e for e in events if e.get("event") in ("ask", "analyze", "search", "alerts", "conflicts")]
    fb = [e for e in events if e.get("event") == "feedback"]
    asks = [e for e in events if e.get("event") == "ask"]
    lat = [e["ms"] for e in asks if isinstance(e.get("ms"), (int, float))]
    return dict(as_of=as_of.isoformat(), users_total=len({e["user"] for e in acts}),
                weekly_active_users=len({e["user"] for e in acts if e.get("day") in week}),
                actions_last_7_days=sum(e.get("day") in week for e in acts), questions_total=len(asks),
                packages_analysed=sum(e.get("event") == "analyze" for e in events),
                feedback_total=len(fb), useful_share=round(sum(bool(e.get("useful")) for e in fb) / len(fb), 3) if fb else None,
                answers_held_back=sum(bool(e.get("held_back")) for e in asks), refusals=sum(bool(e.get("refused")) for e in asks),
                median_answer_ms=round(statistics.median(lat)) if lat else None,
                p95_answer_ms=round(sorted(lat)[max(0, -(-95 * len(lat) // 100) - 1)]) if lat else None)   # nearest rank
