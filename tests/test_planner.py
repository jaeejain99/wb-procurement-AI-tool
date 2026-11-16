"""Context engineering: every kind of question runs the tools it needs, before and independent of the model."""
import json, warnings; warnings.filterwarnings("ignore")
import pytest
from procintel.planner import evidence_pack, route, snippet

CN = {"KE": "Kenya", "GH": "Ghana", "NP": "Nepal", "BD": "Bangladesh", "VN": "Viet Nam", "RW": "Rwanda"}
SECTORS = ["Transportation", "Health", "Education", "Water, Sanitation and Waste Management"]
tools = lambda p: [c[0] for c in p.calls]


@pytest.mark.parametrize("q,must", [
    ("We are about to tender USD 20m of rural road rehabilitation works in Kenya. What should we plan for?", {"forecast_package", "search_documents"}),
    ("What do completion reports say about land acquisition delays in works contracts?", {"search_documents"}),
    ("How long do consultant selections take in Ghana, and what slows them down?", {"forecast_package", "search_documents"}),
    ("Which active projects in Bangladesh show early-warning signs, and why?", {"early_warnings"}),
    ("What is the outcome rating of project P159238, and did its supervision ratings agree?", {"get_project", "search_documents"}),
    ("Why is project P179414 flagged, and what should the team do about it?", {"get_project", "early_warnings", "search_documents"}),
    ("Tell me something useful", {"search_documents"}),                           # anything else: at least a search
])
def test_required_tools_per_question_type(q, must):
    assert must <= set(tools(route(q, CN, SECTORS)))


def test_package_plan_searches_several_risk_themes_and_procurement_sections():
    p = route("We are about to tender USD 20m of rural road rehabilitation works in Kenya. What should we plan for?", CN, SECTORS)
    searches = [a for n, a in p.calls if n == "search_documents"]
    assert len(searches) >= 3 and {"lessons", "procurement"} <= {a["section"] for a in searches} and all(a["country"] == "Kenya" for a in searches)
    fc = next(a for n, a in p.calls if n == "forecast_package")
    assert fc == dict(category="Civil Works", country_code="KE", sector="Transportation", value_usd_m=20.0)


def test_follow_up_inherits_the_package_and_switches_country():
    first = route("We are about to tender USD 20m of rural road rehabilitation works in Kenya. What should we plan for?", CN, SECTORS)
    p = route("What about the same package in Ghana?", CN, SECTORS, prev_plan=first)
    fc = next(a for n, a in p.calls if n == "forecast_package")
    assert fc["country_code"] == "GH" and fc["category"] == "Civil Works" and fc["value_usd_m"] == 20.0


def test_workspace_context_drives_the_forecast():
    ctx = dict(category="Goods", sector="Health", country_code="NP", value_usd_m=5, method="Request for Bids")
    p = route("What should we watch out for?", CN, SECTORS, context=ctx)
    assert ("forecast_package", dict(category="Goods", country_code="NP", sector="Health", procurement_method="Request for Bids", value_usd_m=5)) in p.calls


def test_evidence_pack_is_compact_and_keeps_numbers():
    src = {"ICR_A#1": dict(text="Intro text here about the project. Bid prices came in 35 percent above the cost estimate, forcing re-tendering. "
                                "Other unrelated detail about staffing and training matters. " * 3)}
    res = [("search_documents", dict(query="bid prices above the cost estimate"),
            [dict(id="ICR_A#1", doc_type="ICR", project_id="P1", country="Kenya", section="Lessons", text=src["ICR_A#1"]["text"])])]
    pack = evidence_pack(res, src, "cost estimate")
    assert pack.startswith("[ICR_A#1] ICR P1") and "35 percent above the cost estimate" in pack and len(pack) < 600
    assert len(snippet("x " * 5000, "y")) <= 440


def test_required_tools_run_even_if_the_model_never_calls_one(tmp_path):
    from procintel import Platform
    from procintel.agent import Agent
    from procintel.synth import generate, write_dataset
    write_dataset(generate(seed=3, n_projects=60), str(tmp_path)); plat = Platform.build(str(tmp_path))
    class NeverCallsTools:
        def __init__(self): self.n = 0
        def chat(self, model, messages, tools=None, **kw):
            self.n += 1; return {"content": "Not enough evidence to answer."}
    class Judge:
        def __call__(self, items): return [(True, "ok")] * len(items)
    m = NeverCallsTools()
    r = Agent(plat, client=m, judge=Judge()).ask("We plan USD 12m of civil works roads in Kenya. What should we prepare for?")
    assert {"forecast_package", "search_documents"} <= set(r["tools_run"]) and m.n == 1


def test_model_can_add_one_round_of_tools(tmp_path):
    from procintel import Platform
    from procintel.agent import Agent
    from procintel.synth import generate, write_dataset
    write_dataset(generate(seed=3, n_projects=60), str(tmp_path)); plat = Platform.build(str(tmp_path))
    pid = plat.ds.projects.project_id.iloc[0]
    class AsksForMore:
        def __init__(self): self.n = 0
        def chat(self, model, messages, tools=None, **kw):
            self.n += 1
            if self.n == 1 and tools:
                return {"content": "", "tool_calls": [{"id": "x", "function": {"name": "get_project", "arguments": json.dumps({"project_id": pid})}}]}
            return {"content": "Not enough evidence to answer."}
    m = AsksForMore()
    r = Agent(plat, client=m, judge=None, use_judge=False).ask("What lessons exist on cost estimates?")
    assert any(t.startswith("round 2: get_project") for t in r["trace"]) and m.n == 2
