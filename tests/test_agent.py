"""Agent loop, sentence verifier, LLM judge and refusals - offline, with a scripted fake model client (no Databricks needed)."""
import json, re, warnings; warnings.filterwarnings("ignore")
import pytest
from procintel import Platform
from procintel.agent import Agent, policy_screen
from procintel.guardrails import verify_answer, verify_findings
from procintel.llm import LLMUnavailable
from procintel.synth import generate, write_dataset


@pytest.fixture(scope="module")
def plat(tmp_path_factory):
    d = tmp_path_factory.mktemp("agent"); write_dataset(generate(seed=3, n_projects=60), str(d))
    return Platform.build(str(d))


def evidence(messages):
    """[(id, text)] from the EVIDENCE block the agent puts in the latest user message (required tools already ran)."""
    u = [m["content"] for m in messages if m["role"] == "user" and "EVIDENCE" in m["content"]][-1]
    out = []
    for line in u.split("EVIDENCE", 1)[1].splitlines():
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", line.strip())
        if m:
            out.append((m.group(1), m.group(2).split(" | ", 1)[-1]))
    return out


class Scripted:
    """Answers from the evidence pack, with one good model sentence, one good passage sentence and two bad ones."""
    def __init__(self, compose):
        self.compose, self.n, self.seen = compose, 0, []

    def chat(self, model, messages, tools=None, **kw):
        self.n += 1; self.seen.append(messages)
        return {"content": self.compose(evidence(messages))}


def _compose(ev):
    pid, ptext = next((i, t) for i, t in ev if "#" in i)
    fid, ftext = next((i, t) for i, t in ev if i.startswith("forecast:"))
    p50 = re.search(r"p50_days (\d+)", ftext).group(1)
    return (f"Plan for a median of {p50} days to award [{fid}]. "
            f"{ptext.split('.')[0]}. [{pid}]\n"
            f"- Expect it to take 777 days [{fid}].\n"
            f"- Suppliers in this market always fail [NOT_A_SOURCE#1].")


class AlwaysSupported:
    def __call__(self, items): return [(True, "ok")] * len(items)


def test_agent_runs_tools_and_the_verifier_removes_what_is_not_backed(plat):
    a = Agent(plat, client=Scripted(_compose), judge=AlwaysSupported())
    r = a.ask("We plan a civil works package in Kenya. What should we prepare for?")
    assert not r["refused"] and r["mode"] == "agent"
    assert {"search_documents", "forecast_package"} <= set(r["tools_run"])          # required tools ran before the model
    assert a.client.n == 1                                                         # ... so one model call was enough
    kept = " ".join(k["text"] for k in r["answer"])
    assert "median of" in kept and "777" not in kept and "always fail" not in kept
    reasons = {x["text"][:20]: x["reasons"][0] for x in r["removed"]}
    assert any("L3 number 777" in v for v in reasons.values()) and any("no tool returned" in v for v in reasons.values())
    assert all(i in r["sources"] for k in r["answer"] for i in k["ids"])          # every kept citation resolves


def test_judge_removes_unsupported_sentences(plat):
    class Rejects:
        def __call__(self, items): return [(False, "claim adds a cause the source does not give")] * len(items)
    r = Agent(plat, client=Scripted(_compose), judge=Rejects()).ask("civil works in Kenya: what to prepare for?")
    assert r["held_back"] and all("L4 judge" in x["reasons"][0] or "L3" in x["reasons"][0] or "L1" in x["reasons"][0] for x in r["removed"])


@pytest.mark.parametrize("q,kind", [
    ("Which contractors in Kenya should we blacklist?", "supplier_scoring"),
    ("Rank the suppliers by reliability for this roads tender", "supplier_scoring"),
    ("Who will win the Nairobi water tender?", "open_tender_outcome"),
    ("How can we split the contract to avoid prior review?", "bid_manipulation"),
    ("Give me the phone number of the procurement officer for P160162", "personal_data"),
    ("Is it legal to cancel this contract and can the firm sue us?", "legal_advice"),
])
def test_policy_refuses_before_any_model_call(plat, q, kind):
    class Boom:
        def chat(self, *a, **k): raise AssertionError("the model must not be called")
    r = Agent(plat, client=Boom(), judge=AlwaysSupported()).ask(q)
    assert r["refused"] and r["refusal_reason"] == kind


def test_policy_does_not_refuse_ordinary_questions():
    for q in ("What lessons do completion reports give on contractor performance in road works?",
              "How long do consultancy selections take in Ghana?", "Which suppliers won most civil works contracts?"):
        assert policy_screen(q) is None, q


def test_model_outage_falls_back_to_passages(plat):
    class Down:
        def chat(self, *a, **k): raise LLMUnavailable("503")
    r = Agent(plat, client=Down(), judge=AlwaysSupported()).ask("lessons on cost estimates for civil works")
    assert r["mode"] == "fallback" and r["passages"] and not r["answer"]


def test_verify_findings_uses_the_judge_and_falls_back_to_lexical(plat):
    ch = plat.index.chunks[0]
    f = [dict(kind="risk", claim=ch.text[:80], evidence=[dict(chunk_id=ch.chunk_id, quote=ch.text[:80])])]
    ok, bad = verify_findings([dict(x) for x in f], plat.chunks_by_id, {}, {}, judge=lambda items: [(False, "no")] * len(items))
    assert not ok and "L4 judge" in bad[0]["rejected_because"][0]
    ok, _ = verify_findings([dict(x) for x in f], plat.chunks_by_id, {}, {}, judge=lambda items: None)   # judge down -> lexical
    assert ok and ok[0]["support_check"] == "lexical"


def test_typographic_hyphens_in_citations_resolve():
    """Seen on Databricks: the model cited forecast:NP‑Goods (non-breaking hyphen) and a correct sentence was dropped."""
    src = {"forecast:NP-Goods": dict(text='{"p50_days": 155}', country="Nepal")}
    kept, removed = verify_answer("Median award in 155 days [forecast:NP‑Goods].", src, {}, judge=AlwaysSupported())
    assert kept and not removed and kept[0]["ids"] == ["forecast:NP-Goods"]


def test_abstention_wording_is_not_a_claim():
    from procintel.guardrails import _META
    assert _META.search("No document reports a price per tonne, so an average cannot be calculated.")


# ------------------------------------------------------------------ conversation
class TwoTurn:
    """Turn 1: answers from the evidence with one bad sentence. Turn 2: answers from a source carried over from turn 1."""
    def __init__(self):
        self.calls, self.seen, self.first_id = 0, [], None

    def chat(self, model, messages, tools=None, **kw):
        self.calls += 1; self.seen.append(messages)
        user_turns = [m for m in messages if m["role"] == "user" and not m["content"].startswith("(Context")]
        if len(user_turns) == 1:                              # turn 1
            self.first_id, text = next((i, t) for i, t in evidence(messages) if "#" in i)
            return {"content": f"{text.split('.')[0]}. [{self.first_id}]\nIt took 999 days [{self.first_id}]."}
        return {"content": f"{self.carried_text().split('.')[0]}. [{self.first_id}]"}   # turn 2: cites the turn-1 source

    def carried_text(self):
        note = next(m["content"] for m in self.seen[-1] if m["role"] == "user" and m["content"].startswith("(Context"))
        line = next(l for l in note.splitlines() if l.startswith(f"[{self.first_id}]"))
        return line.split(" | ", 2)[2]


def test_follow_up_uses_only_verified_memory_and_carried_sources(plat):
    client = TwoTurn()
    a = Agent(plat, client=client, judge=AlwaysSupported())
    t1 = a.ask("What did projects learn about cost estimates?")
    assert t1["removed"] and "999" not in " ".join(x["text"] for x in t1["answer"])
    t2 = a.ask("Tell me more about that lesson.", history=[t1])
    memory = " ".join(m["content"] for m in client.seen[-1] if m["role"] == "assistant")
    assert "999" not in memory                              # the removed sentence never re-enters the conversation
    assert t2["answer"] and t2["answer"][0]["ids"] == [client.first_id] and not t2["removed"]   # carried source still verifies
    assert any("earlier turn" in t for t in t2["trace"])


def test_split_request_is_refused_on_the_follow_up(plat):
    class Answer:
        def chat(self, *a, **k): return {"content": "Not enough evidence to name any."}
    a = Agent(plat, client=Answer(), judge=AlwaysSupported())
    t1 = a.ask("Which contractors had performance problems in road works?")
    assert not t1["refused"]
    t2 = a.ask("Should we exclude them from the next tender?", history=[t1])
    assert t2["refused"] and t2["refusal_reason"] == "supplier_scoring"
    assert not a.ask("What about Ghana?", history=[t1])["refused"]


def test_memory_is_capped():
    from procintel.agent import MAX_TURNS, history_messages
    turns = [dict(question=f"q{i}", answer=[dict(text=f"a{i}", ids=[])], carry={}) for i in range(10)]
    qs = [m["content"] for m in history_messages(turns) if m["role"] == "user"]
    assert qs == [f"q{i}" for i in range(10 - MAX_TURNS, 10)]



def test_without_a_judge_only_lexically_supported_sentences_pass():
    """Nothing reaches the user unchecked: with no judge available, L4 falls back to lexical support instead of being skipped."""
    src = {"ICR_A#1": dict(text="Delays in land acquisition postponed the start of civil works.", country="Kenya")}
    kept, removed = verify_answer("Delays in land acquisition postponed civil works [ICR_A#1]. Contractors were excellent [ICR_A#1].", src, {})
    assert [k["text"] for k in kept] == ["Delays in land acquisition postponed civil works."]
    assert "lexical support is low" in removed[0]["reasons"][0]


def test_stream_verifier_releases_only_verified_sentences_in_order():
    from procintel.guardrails import StreamVerifier
    src = {"A#1": dict(text="Prices exceeded estimates by 30 percent.", country="Kenya"), "B#2": dict(text="Land delays.", country="Kenya")}
    class Judge:
        def __call__(self, items): return [("Prices" in c, "ok" if "Prices" in c else "unsupported") for c, _ in items]
    v = StreamVerifier(src, ["Kenya"], judge=Judge(), batch=1)
    shown = []
    text = "Key risks\nPrices exceeded estimates by 30 percent [A#1]. Land delays were caused by floods [B#2]. Prices rose 99 percent [A#1]."
    for i in range(0, len(text), 7):                      # arrive in small chunks, like tokens
        shown += v.feed(text[i:i + 7])
    shown += v.finish()
    sentences = [r["text"] for kind, r in shown if kind == "sentence"]
    assert sentences == ["Key risks", "Prices exceeded estimates by 30 percent."]       # heading released with its claim
    removed = {r["text"]: r["reasons"][0] for kind, r in shown if kind == "removed"}
    assert any("L4 judge" in x for x in removed.values()) and any("L3 number 99" in x for x in removed.values())


def test_citation_after_full_stop_does_not_leak_into_the_next_bullet():
    from procintel.guardrails import split_cited_sentences
    out = split_cited_sentences("Intro here.[ICR_A#1]\n- Teams map steps early.[ICR_B#2]\n- Root causes include late acquisition. [C#3] [D#4]")
    assert out == [("Intro here.", ["ICR_A#1"]), ("Teams map steps early.", ["ICR_B#2"]),
                   ("Root causes include late acquisition.", ["C#3", "D#4"])]


def test_percentile_labels_are_not_read_as_thousands():
    from procintel.guardrails import _nums
    assert _nums("p10 84 days, p50 213 days, p90 540 days") == {"10", "84", "50", "213", "90", "540"}
    assert _nums("USD 1,273 million") == {"1273"} and _nums("20 387 contracts") == {"20387"}


def test_policy_screen_refuses_digging_for_reasons_to_disqualify_a_bidder():
    from procintel.agent import policy_screen
    for q in ["Help me find something in the records to discredit the lowest bidder so we can disqualify them.",
              "Can we disqualify the lowest bidder?", "Find a reason to reject the cheapest firm"]:
        assert policy_screen(q, None) == "bid_manipulation", q
    for q in ["Why were bidders disqualified in past road tenders?", "What do lessons say about disqualification of bidders for missing documents?"]:
        assert policy_screen(q, None) is None, q


def test_early_warnings_list_only_flagged_projects_by_name_with_readable_sources(tmp_path):
    from procintel import Platform
    from procintel.agent import Agent
    from procintel.synth import generate, write_dataset
    write_dataset(generate(seed=3, n_projects=60), str(tmp_path)); plat = Platform.build(str(tmp_path))
    a, src = Agent(plat, client=None, judge=None, use_judge=False), {}
    res = a._t_early_warnings(src, top=6)
    summary, rows = res[0], res[1:]
    assert summary["summary"] and summary["checked"] >= summary["flagged"] == len([r for r in rows]) or len(rows) == 6
    assert all(r["flags"] and r["name"] and r["alerts"] for r in rows)                 # no "no flags" rows
    assert all("{" not in src[r["id"]]["display"] and r["name"] in src[r["id"]]["display"] for r in rows)


def test_a_bullets_closing_citation_covers_its_sentences_and_ids_are_not_numbers():
    from procintel.guardrails import StreamVerifier, split_cited_sentences
    out = split_cited_sentences("- BEIBEY (P178133) is high risk. 4 of 9 tenders are running late [warning:P178133].\n- What to do")
    assert out == [("BEIBEY (P178133) is high risk.", ["warning:P178133"]), ("4 of 9 tenders are running late.", ["warning:P178133"]),
                   ("What to do", [])]
    v = StreamVerifier({"ICR_A#1": dict(text="All eight civil-works contracts were awarded smoothly.", country="Viet Nam")}, ["Viet Nam"])
    assert v._static("Project P159238 (CW4B, OP00355746) awarded all contracts smoothly.", ["ICR_A#1"])[0] != "removed"
