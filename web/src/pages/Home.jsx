import { Stat, go } from "../components";

const GUIDE = [
  {
    key: "package", icon: "📦", title: "Package workspace",
    what: "Plan a new procurement package with evidence from past World Bank projects: a realistic schedule, which procurement method to use, and the risks to expect.",
    how: [
      "Fill in the package: category, sector, country, value and the method you are considering.",
      "Press Analyse. Step 1 shows the schedule range and compares procurement methods.",
      "Step 2 lists risks from past projects, each with a quoted lesson and a mitigation.",
      "Use “Ask about this package” for follow-up questions.",
    ],
    who: "For procurement specialists preparing a package (Planning & strategy, packaging).",
  },
  {
    key: "portfolio", icon: "📈", title: "Portfolio monitor",
    what: "See which active projects need attention and why: falling supervision ratings, tenders open longer than expected, and disagreements between supervision and completion reports.",
    how: [
      "Optionally filter by country; projects are ranked by risk.",
      "Read the flags on each row: every flag says what triggered it.",
      "Open a project for its rating history, open tenders against their expected duration, and its lessons.",
      "Use “Ask why” to explore a flagged project in conversation.",
    ],
    who: "For task team leaders and managers (Execution & monitoring).",
  },
  {
    key: "ask", icon: "💬", title: "Ask",
    what: "Ask anything in plain words and follow up. An agent decides which tools to use (document search, the schedule model, project records, early warnings) and answers with citations.",
    how: [
      "Type a question, or start from a suggestion or a package.",
      "Ask follow-ups such as “What about Ghana?”: the conversation remembers the last few answers.",
      "Click a numbered citation to read the exact passage, or open Sources to see everything the answer is based on.",
      "Start a new conversation when you change topic.",
    ],
    who: "For anyone, especially questions the other tabs don't anticipate.",
  },
  {
    key: "library", icon: "📚", title: "Evidence library",
    what: "Search the project documents directly (completion reports, appraisal documents, status reports) and read passages in context. Nothing is generated here, only quotes.",
    how: [
      "Search with your own words; filter by document type, section (e.g. lessons, procurement) or country.",
      "Open a passage to read it in the full report at its page, or open the original PDF at that page.",
      "Citations everywhere else in Tenderwise open here at the exact passage.",
      "Mark passages useful or not: feedback tunes what is shown first.",
    ],
    who: "For checking a source, or finding precedent yourself.",
  },
];

export default function Home({ meta }) {
  const s = meta.summary;
  return (
    <>
      <section className="hero">
        <h1>Procurement Intelligence Platform</h1>
        <p>Tenderwise links contract awards, procurement notices and project records with the reports that explain why procurements
          went well or badly, so you can plan packages, watch projects and learn from past lessons, with every statement traced to its source.</p>
        <div className="stats" style={{ marginTop: 18 }}>
          <Stat label="Contracts" value={s.contracts.toLocaleString()} sub={`${s.countries} countries`} />
          <Stat label="Project reports" value={s.documents.toLocaleString()} />
          <Stat label="Active projects" value={s.active} sub={`${s.projects} projects in total`} />
          <Stat label="Tenders still open" value={s.open_tenders} sub="tracked against expected duration" />
        </div>
      </section>

      <div className="spread" style={{ margin: "6px 0 12px" }}>
        <span className="tiny">Start with the Package workspace if you are preparing a tender.</span>
      </div>
      <div className="grid-cards">
        {GUIDE.map((g, i) => (
          <article key={g.key} className="card tabcard">
            <div className="head"><span className="icon" aria-hidden>{g.icon}</span>
              <div><div className="tiny">Tab {i + 1}</div><h2>{g.title}</h2></div></div>
            <p className="for">{g.what}</p>
            <div className="small" style={{ fontWeight: 600 }}>How to use it</div>
            <ol>{g.how.map((h, k) => <li key={k}>{h}</li>)}</ol>
            <div className="who">{g.who}</div>
            <button className="btn primary" style={{ marginTop: 12, alignSelf: "flex-start" }} onClick={() => go(`/${g.key}`)}>Open {g.title} →</button>
          </article>
        ))}
      </div>
    </>
  );
}
