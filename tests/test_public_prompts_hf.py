import json
import pytest
from nar_guardrails.datasets.public_prompts import (load_neuralchemy, load_prodnull, load_ipi_arena, load_bells,
                                                     load_llmail, load_pliny, load_simsonsun, load_mindgard,
                                                     load_wildchat, HF_SOURCES)


def test_neuralchemy_filters_old_sources_and_maps_categories():
    rows = [{"text": "ignore prev", "label": 1, "category": "direct_injection", "source": "original", "tags": []},
            {"text": "harm", "label": 1, "category": "jailbreak", "source": "harmbench", "tags": []},
            {"text": "repeat above", "label": 1, "category": "prompt_leaking", "source": "original", "tags": []},
            {"text": "hi", "label": 0, "category": "benign", "source": "hand-crafted", "tags": ["hard_negative"]},
            {"text": "doc attack", "label": 1, "category": "indirect_injection", "source": "original", "tags": []}]
    items = load_neuralchemy(rows, cap=None, seed=0)
    assert [i.scenario for i in items] == ["D1", "D2", "B2", "I1"]
    assert all(i.source == "neuralchemy" and i.meta["published"] == "2026-01" for i in items)


def test_prodnull_balanced_cap_and_scenarios():
    rows = [{"text": f"README {k}", "label": k % 2} for k in range(20)]
    items = load_prodnull(rows, cap=6, seed=0)
    assert len(items) == 6 and sum(i.label for i in items) == 3
    assert {i.scenario for i in items} == {"I5", "B3"}


def test_ipi_arena_all_rows_positive():
    items = load_ipi_arena([{"behavior_id": "book-hotel", "attack": "<system>do x</system>"}], cap=None, seed=0)
    assert items[0].id.startswith("ipi_arena:") and items[0].meta["behavior"] == "book-hotel" and items[0].label == 1 and items[0].scenario == "I1"


def test_bells_benign_origin_is_label_zero():
    rows = [{"id": "h1", "prompt": "p", "category": "Cyber", "origin": "generated", "jailbreak_prompt": "b64 of harm", "technique": "to_base64", "technique_type": "encoding_cyphering"},
            {"id": "b1", "prompt": "p", "category": "Benign", "origin": "handcrafted", "jailbreak_prompt": "b64 of recipe", "technique": "to_base64", "technique_type": "encoding_cyphering"}]
    items = load_bells(rows, cap=500, seed=0)
    by = {i.id: i for i in items}
    assert by["bells:h1:to_base64"].label == 1 and by["bells:h1:to_base64"].scenario == "J2"
    assert by["bells:b1:to_base64"].label == 0 and by["bells:b1:to_base64"].scenario == "B2"


def test_llmail_keeps_only_successful_and_builds_email_text():
    ok = {"subject": "Hi", "body": "please forward", "scenario": "level2a", "output": json.dumps({"email.retrieved": True, "defense.undetected": True, "exfil.sent": False})}
    bad = {"subject": "x", "body": "y", "scenario": "level1", "output": {"email.retrieved": True, "defense.undetected": False, "exfil.sent": False}}
    items = load_llmail([ok, bad], cap=None, seed=0)
    assert len(items) == 1 and items[0].text == "Email subject: Hi\n\nplease forward" and items[0].scenario == "I2"


def test_pliny_filters_passed_text_challenges():
    rows = [{"submission_id": "s1", "challenge_slug": "pliny_9_challenge", "passed": True,
             "messages": [{"role": "user", "content": "trick 1"}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "trick 2"}]},
            {"submission_id": "s2", "challenge_slug": "pliny_6_challenge", "passed": True, "messages": [{"role": "user", "content": "img"}]},
            {"submission_id": "s3", "challenge_slug": "pliny_2_challenge", "passed": False, "messages": [{"role": "user", "content": "x"}]}]
    items = load_pliny(rows, cap=None, seed=0)
    assert len(items) == 1 and items[0].text == "trick 1\n\ntrick 2" and items[0].scenario == "J2"
    assert items[0].meta["slice"] == "promptguard-evasion"


def test_simsonsun_and_mindgard_and_wildchat():
    s = load_simsonsun([{"Source": "reddit", "Prompt": "dan prompt"}], cap=None, seed=0)
    assert s[0].label == 1 and s[0].scenario == "J1"
    m = load_mindgard([{"original_prompt": "ignore", "modified_prompt": "i g n o r e", "attack_name": "spaces"}], cap=None, seed=0)
    assert m[0].text == "i g n o r e" and m[0].scenario == "J2" and m[0].meta["attack_name"] == "spaces"
    w = load_wildchat([{"language": "English", "toxic": False, "conversation": [{"role": "user", "content": "Plan a 3-day trip to Rome for me please"}]},
                       {"language": "English", "toxic": True, "conversation": [{"role": "user", "content": "bad bad bad bad bad bad"}]},
                       {"language": "German", "toxic": False, "conversation": [{"role": "user", "content": "Plane eine Reise nach Rom für mich"}]}], cap=10, seed=0)
    assert len(w) == 1 and w[0].label == 0 and w[0].scenario == "B1"


def test_hf_sources_registry():
    assert set(HF_SOURCES) == {"neuralchemy", "prodnull", "ipi_arena", "bells", "llmail", "pliny", "simsonsun", "mindgard", "wildchat"}
    for s in HF_SOURCES.values():
        assert s["published"] >= "2025-01" and callable(s["loader"]) and "dataset" in s


def _bells(i, tech="to_base64", cat="Cyber", tt="encoding_cyphering"):
    return {"id": i, "category": cat, "jailbreak_prompt": f"{i}-{tech}-{cat}", "technique": tech, "technique_type": tt}


def test_bells_ids_unique_for_shared_base_id_and_technique():
    items = load_bells([_bells("x", "a"), _bells("x", "b")], cap=10, seed=0)
    assert {i.id for i in items} == {"bells:x:a", "bells:x:b"}
    rows = [dict(_bells("x", "a"), jailbreak_prompt="p1"), dict(_bells("x", "a"), jailbreak_prompt="p2")]
    assert len({i.id for i in load_bells(rows, cap=10, seed=0)}) == 2


def test_bells_benign_capped_and_scenario_families():
    rows = [_bells(f"b{k}", cat="Benign") for k in range(10)]
    assert len(load_bells(rows, cap=3, seed=0)) == 3
    fam = load_bells([_bells("c", tt="cognitive_psychological"), _bells("f", tt="few_shot_hacking"),
                      _bells("d", tt="distract_and_persuade"), _bells("e", tt="low_resource_language")], cap=10, seed=0)
    assert {i.id.split(":")[1]: i.scenario for i in fam} == {"c": "J1", "f": "J1", "d": "J1", "e": "J2"}


def test_index_free_ids_stable_across_cap_and_dedupe():
    rows = [{"Source": "r", "Prompt": "same"}, {"Source": "r", "Prompt": "same"}, {"Source": "r", "Prompt": "other"}]
    full = load_simsonsun(rows, cap=None, seed=0)
    assert len(full) == 2
    one = load_simsonsun(rows, cap=1, seed=0)
    assert len(one) == 1 and one[0].id in {i.id for i in full}
    same = {i.text: i.id for i in full}
    assert one[0].id == same[one[0].text]


def test_mindgard_exact_column_wins_and_missing_raises_and_encoded_flag():
    row = {"original_modified_flag": "no", "modified_prompt": "evaded", "attack_name": "Emoji Smuggling"}
    m = load_mindgard([row], cap=None, seed=0)
    assert m[0].text == "evaded" and m[0].meta["encoded"] is True
    with pytest.raises(KeyError):
        load_mindgard([{"original_prompt": "x", "attack_name": "a"}], cap=None, seed=0)


def test_llmail_skips_invalid_json_output():
    row = {"subject": "s", "body": "b", "output": "{not json"}
    assert load_llmail([row], cap=None, seed=0) == []
