// Thin client for the FastAPI backend. Errors come back as {detail}; turn them into readable messages.
async function request(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  } catch {
    throw new Error("Cannot reach the Tenderwise server. Check your connection and try again.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    let msg = body.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return body;
}

const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== "")).toString();

export const api = {
  health: () => request("api/health"),
  meta: () => request("api/meta"),
  analyze: (pkg) => request("api/package/analyze", { method: "POST", body: JSON.stringify(pkg) }),
  portfolio: (p) => request(`api/portfolio?${qs(p)}`),
  project: (pid) => request(`api/project/${encodeURIComponent(pid)}`),
  ask: (b) => request("api/ask", { method: "POST", body: JSON.stringify(b) }),
  // Server-sent events over POST: status, tool, sentence (verified only), removed, done. Returns the final result.
  askStream: async (b, onEvent) => {
    const res = await fetch("api/ask/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({}));
      throw Object.assign(new Error(body.detail || `Request failed (${res.status})`), { status: res.status });
    }
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "", result = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, k); buf = buf.slice(k + 2);
        const line = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.type === "error") throw new Error(ev.detail);
        if (ev.type === "done") result = ev.result;
        onEvent(ev);
      }
    }
    if (!result) throw new Error("The answer stream ended early.");
    return result;
  },
  search: (p) => request(`api/search?${qs(p)}`),
  passage: (id) => request(`api/passage/${encodeURIComponent(id)}`),
  document: (id) => request(`api/document/${encodeURIComponent(id)}`),
  feedback: (b) => request("api/feedback", { method: "POST", body: JSON.stringify(b) }),
};
