"""Chat client for Databricks Model Serving (Foundation Model API, OpenAI-compatible chat + tool calling). All-Databricks.

Auth, first that works:
  1 DATABRICKS_HOST + DATABRICKS_TOKEN
  2 databricks-sdk default config (a Databricks App's service principal, a notebook, or ~/.databrickscfg)
  3 the Databricks CLI login (`databricks auth token`) - local development
Models (tested on the Free Edition workspace, 2026-11-03): gpt-oss-120b is the only served model that reliably runs a
multi-tool plan AND cites every sentence; llama-3.3-70b is a different family for the judge (less correlated mistakes)."""
from __future__ import annotations
import configparser, json, os, shutil, ssl, subprocess, time, urllib.error, urllib.request

AGENT_MODEL = os.getenv("PROCINTEL_AGENT_MODEL", "databricks-gpt-oss-120b")
JUDGE_MODEL = os.getenv("PROCINTEL_JUDGE_MODEL", "databricks-meta-llama-3-3-70b-instruct")


class LLMUnavailable(RuntimeError):
    pass


class OfflineChat:
    """The client used when PROCINTEL_OFFLINE is set: every model call fails as 'unavailable', so the agent takes its
    deterministic fallback (retrieved passages, no generated text) even on a machine that could reach a model."""
    def chat(self, *a, **k):
        raise LLMUnavailable("offline mode (PROCINTEL_OFFLINE)")


def _ssl():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class DatabricksChat:
    def __init__(self, host=None, token=None):
        import threading
        self._host, self._token, self._exp, self._sdk = host, token, 0.0, None
        self._lock = threading.Lock()          # parallel tool calls must not each fetch a token (a cold-start stampede)
        self.ctx = _ssl()
        self.calls = 0

    def _auth(self):
        with self._lock:
            return self._auth_unlocked()

    def _auth_unlocked(self):
        if os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
            return os.environ["DATABRICKS_HOST"].rstrip("/"), {"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}
        try:
            from databricks.sdk.core import Config
            self._sdk = self._sdk or Config()
            return self._sdk.host.rstrip("/"), self._sdk.authenticate()
        except Exception:
            pass
        if self._token and time.time() < self._exp:
            return self._host, {"Authorization": f"Bearer {self._token}"}
        exe = shutil.which("databricks")
        cfg = configparser.ConfigParser(); cfg.read(os.path.expanduser("~/.databrickscfg"))
        host = self._host or next((cfg[s].get("host") for s in cfg.sections() if cfg[s].get("host")), None) or cfg.defaults().get("host")
        if not exe or not host:
            raise LLMUnavailable("no Databricks credentials (set DATABRICKS_HOST/TOKEN, or log in with the Databricks CLI)")
        out = subprocess.run([exe, "auth", "token", "--host", host, "-o", "json"], capture_output=True, text=True, timeout=60)
        if out.returncode:
            raise LLMUnavailable(f"databricks auth token failed: {out.stderr.strip()[:200]}")
        self._host, self._token, self._exp = host.rstrip("/"), json.loads(out.stdout)["access_token"], time.time() + 1800
        return self._host, {"Authorization": f"Bearer {self._token}"}

    def chat(self, model, messages, tools=None, max_tokens=900, temperature=0.0, retries=4) -> dict:
        """One chat completion -> the assistant message dict (content, tool_calls). Retries rate limits / transient 5xx."""
        host, headers = self._auth()
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        data = json.dumps(body).encode()
        for i in range(retries):
            req = urllib.request.Request(f"{host}/serving-endpoints/{model}/invocations", data=data, method="POST",
                                         headers={**headers, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180, context=self.ctx) as r:
                    self.calls += 1
                    return json.loads(r.read())["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                    time.sleep(2 * 2 ** i); continue
                raise LLMUnavailable(f"{model}: HTTP {e.code} {e.read()[:200]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if i < retries - 1:
                    time.sleep(2 * 2 ** i); continue
                raise LLMUnavailable(f"{model}: {e}") from e
        raise LLMUnavailable(f"{model}: retries exhausted")


_SHARED = None


def shared_client() -> DatabricksChat:
    """One client (one token, one lock) for the agent, the judge and Vector Search in this process."""
    global _SHARED
    if _SHARED is None:
        _SHARED = DatabricksChat()
    return _SHARED


def chat_stream(client: DatabricksChat, model, messages, tools=None, max_tokens=600, temperature=0.0, retries=3, reasoning_effort=None):
    """Streaming chat completion (OpenAI-style server-sent events, verified on the workspace for gpt-oss-120b, incl. streamed
    tool calls). Yields ("text", delta) as content arrives, then ("end", {"content": full_text, "tool_calls": [...]}).
    Retries only before the first byte arrives, so no text is ever duplicated."""
    host, headers = client._auth()
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    if reasoning_effort:                       # gpt-oss: less hidden thinking before the first token (accepted by the endpoint)
        body["reasoning_effort"] = reasoning_effort
    data = json.dumps(body).encode()
    for i in range(retries):
        req = urllib.request.Request(f"{host}/serving-endpoints/{model}/invocations", data=data, method="POST",
                                     headers={**headers, "Content-Type": "application/json", "Accept": "text/event-stream"})
        try:
            resp = urllib.request.urlopen(req, timeout=180, context=client.ctx)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(2 * 2 ** i); continue
            raise LLMUnavailable(f"{model}: HTTP {e.code} {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if i < retries - 1:
                time.sleep(2 * 2 ** i); continue
            raise LLMUnavailable(f"{model}: {e}") from e
    client.calls += 1
    content, calls = [], {}
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = (json.loads(payload).get("choices") or [{}])[0].get("delta") or {}
            except ValueError:
                continue
            c = delta.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type", "text") == "text")
            if c:
                content.append(c)
                yield ("text", c)
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
    yield ("end", {"content": "".join(content), "tool_calls": [calls[k] for k in sorted(calls)]})


def text_of(message) -> str:
    """Assistant content as plain text (some models return a list of content parts)."""
    c = message.get("content") or ""
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type", "text") == "text")
    return c
