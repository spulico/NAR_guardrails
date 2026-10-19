import json
from pathlib import Path
import pytest
from nar_guardrails.schema import read_items
from nar_guardrails.datasets.build_skills import build_skills

ROWS = [{"benchmark_id": f"MSB-{k:04d}", "label": k % 2, "skill_text": f"---\nname: s{k}\n---\nbody {k}", "public_skill_text": None,
         "text_available": True, "source_id": "SRC001", "attack_category": "Execution/Code Delivery" if k % 2 else None} for k in range(10)]
SPLITS = {"MSD": [{"benchmark_id": f"MSB-{k:04d}", "split": ["train", "val", "test"][k % 3]} for k in range(10)],
          "SD": [{"benchmark_id": f"MSB-{k:04d}", "split": "train" if k < 7 else "test"} for k in range(10)]}


def _fixtures(tmp_path):
    f = tmp_path / "skillspector" / "tests" / "fixtures" / "safe_skill"
    f.mkdir(parents=True)
    (f / "SKILL.md").write_text("---\nname: safe\n---\nSafe.")
    return tmp_path / "skillspector"


def test_build_skills_writes_items_manifest_and_counts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("nar_guardrails.datasets.build_skills.load_skillspector_fixtures",
                        lambda repo, labels=None: [])
    out = tmp_path / "skills.jsonl"
    m = build_skills(tmp_path / "raw", out, strict=False, rows=ROWS, split_tables=SPLITS, fixtures_repo=_fixtures(tmp_path))
    items = read_items(out)
    assert m["counts"]["malicious_skill_bench"] == 10 and m["counts"]["curated_skills"] >= 40
    assert {i.source for i in items} >= {"malicious_skill_bench", "curated_skills"}
    assert json.loads(out.with_suffix(".build.json").read_text())["counts"] == m["counts"]
    assert m["protocol_counts"]["MSD"]["train"] == 4 and m["protocol_counts"]["SD"]["test"] == 3


def test_build_skills_keeps_excluded_as_its_own_protocol_bucket(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("nar_guardrails.datasets.build_skills.load_skillspector_fixtures", lambda repo, labels=None: [])
    splits = {"MSD": SPLITS["MSD"], "SD": [dict(r, split="excluded") if r["benchmark_id"] == "MSB-0009" else r for r in SPLITS["SD"]]}
    m = build_skills(tmp_path / "raw", tmp_path / "s.jsonl", strict=False, rows=ROWS, split_tables=splits, fixtures_repo=_fixtures(tmp_path))
    assert m["protocol_counts"]["SD"]["excluded"] == 1 and m["n_items"] == len(read_items(tmp_path / "s.jsonl"))


def test_build_skills_fails_on_overlap(tmp_path: Path, monkeypatch):
    from nar_guardrails.datasets import curated
    leak = curated.load_curated_skills()[0].text
    rows = ROWS + [{"benchmark_id": "MSB-LEAK", "label": 1, "skill_text": leak, "public_skill_text": None, "text_available": True,
                    "source_id": "SRC001", "attack_category": "x"}]
    splits = {"MSD": SPLITS["MSD"] + [{"benchmark_id": "MSB-LEAK", "split": "train"}], "SD": SPLITS["SD"] + [{"benchmark_id": "MSB-LEAK", "split": "train"}]}
    monkeypatch.setattr("nar_guardrails.datasets.build_skills.load_skillspector_fixtures", lambda repo, labels=None: [])
    with pytest.raises(ValueError):
        build_skills(tmp_path / "raw", tmp_path / "s.jsonl", strict=False, rows=rows, split_tables=splits, fixtures_repo=_fixtures(tmp_path))


def test_build_skills_manifest_records_sources(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("nar_guardrails.datasets.build_skills.load_skillspector_fixtures", lambda repo, labels=None: [])
    repo = _fixtures(tmp_path)
    (repo / ".nar_commit").write_text("5ee5ee\n")
    m = build_skills(tmp_path / "raw", tmp_path / "s.jsonl", strict=False, rows=ROWS, split_tables=SPLITS, fixtures_repo=repo)
    assert m["sources"]["malicious_skill_bench"] == {"kind": "hf", "dataset": "ProtectSkills/MaliciousSkillBench",
                                                     "config": "primary", "split": "train"}
    assert m["sources"]["skillspector_fixtures"]["commit"] == "5ee5ee"
