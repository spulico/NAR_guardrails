from collections import Counter
from nar_guardrails.schema import text_sha
from nar_guardrails.datasets.curated import load_curated_prompts, CURATED_PROMPT_COUNTS, SCENARIO_LABEL

FRAMED = {"B3", "I1", "I2", "I3", "I4"}
FRAMING_PREFIXES = ("Tool result", "Email", "Web page", "Document", "Tool description", "Message from agent",
                    "Search result", "Calendar", "Ticket", "File", "API response", "MCP tool")


def test_curated_counts_labels_and_uniqueness():
    items = load_curated_prompts()
    assert Counter(i.scenario for i in items) == CURATED_PROMPT_COUNTS
    assert len(items) == 200
    assert all(i.label == SCENARIO_LABEL[i.scenario] for i in items)
    assert len({i.id for i in items}) == 200
    assert len({text_sha(i.text) for i in items}) == 200
    assert all(len(i.text) >= 20 for i in items)
    assert all(i.source == "curated" and i.split == "test" and i.track == "prompts" for i in items)
    for i in items:
        if i.scenario in FRAMED:
            assert i.text.startswith(FRAMING_PREFIXES), f"{i.id} needs an agent-context framing line"


def test_ids_follow_scenario_pattern():
    for i in load_curated_prompts():
        code, num = i.id.split(":")[1].split("-")
        assert code == i.scenario and num.isdigit()
