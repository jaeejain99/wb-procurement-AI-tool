import { useEffect, useState } from "react";
import { api } from "./api";
import { Spinner } from "./components";
import Home from "./pages/Home";
import PackageWorkspace from "./pages/PackageWorkspace";
import PortfolioMonitor from "./pages/PortfolioMonitor";
import Ask from "./pages/Ask";
import EvidenceLibrary from "./pages/EvidenceLibrary";

export const TABS = [
  { key: "package", label: "Package workspace", icon: "📦" },
  { key: "portfolio", label: "Portfolio monitor", icon: "📈" },
  { key: "ask", label: "Ask", icon: "💬" },
  { key: "library", label: "Evidence library", icon: "📚" },
];

function useRoute() {
  const parse = () => {
    const h = window.location.hash.replace(/^#\/?/, "");
    const [path, q = ""] = h.split("?");
    return { page: path || "home", params: Object.fromEntries(new URLSearchParams(q)) };
  };
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const on = () => { setRoute(parse()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

// The server loads data and models in the background; wait for /api/health before showing pages.
function useReady() {
  const [health, setHealth] = useState({ status: "loading" });
  const [meta, setMeta] = useState(null);
  useEffect(() => {
    let stop = false, timer;
    const poll = async () => {
      try {
        const h = await api.health(); if (stop) return; setHealth(h);
        if (h.status === "ready") { setMeta(await api.meta()); return; }
        if (h.status === "error") return;
      } catch (e) { if (!stop) setHealth({ status: "unreachable", error: e.message }); }
      timer = setTimeout(poll, 2000);
    };
    poll();
    return () => { stop = true; clearTimeout(timer); };
  }, []);
  return { health, meta };
}

export default function App() {
  const { page, params } = useRoute();
  const { health, meta } = useReady();
  const [pkg, setPkg] = useState(null);          // the package analysed in the workspace, carried into Ask

  let body;
  if (!meta) {
    body = (
      <div className="loading-screen"><div>
        {health.status === "error" ? <><h2>Tenderwise could not start</h2><p className="err">{health.error}</p></>
          : health.status === "unreachable" ? <><h2>Server unreachable</h2><p className="small">{health.error} Retrying…</p></>
          : <><Spinner /><h2 style={{ marginBottom: 4 }}>Loading data and models…</h2>
              <p className="small">Contracts, project documents, the survival model and the search index load on start-up (about 30 s).</p></>}
      </div></div>
    );
  } else {
    const props = { meta, params, pkg, setPkg };
    body = { home: <Home {...props} />, package: <PackageWorkspace {...props} />, portfolio: <PortfolioMonitor {...props} />,
             ask: <Ask {...props} />, library: <EvidenceLibrary {...props} /> }[page] || <Home {...props} />;
  }

  return (
    <>
      <header className="topbar"><div className="topbar-inner">
        <a className="brand" href="#/"><span className="brand-mark">T</span>Tenderwise</a>
        <nav className="nav" aria-label="Main">
          {TABS.map((t, i) => <a key={t.key} href={`#/${t.key}`} className={page === t.key ? "on" : ""} aria-current={page === t.key ? "page" : undefined}>
            <span className="num">{i + 1}</span>{t.label}</a>)}
        </nav>
        {meta && <span className="status-pill" title={meta.retrieval}>● {meta.summary.countries} countries · {meta.summary.documents} reports</span>}
      </div></header>
      <main className="page">{body}</main>
    </>
  );
}
