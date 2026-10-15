from pathlib import Path
import yaml
from nar_guardrails.schema import Item
from nar_guardrails.detectors.policy import load_policy, pick_examples


def _write(tmp_path, track, name="shared"):
    d = tmp_path / track
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(yaml.safe_dump(
        {"question": "Is it bad?", "malicious": "bad things", "benign": "good things"}))


def test_load_policy_reads_fields_and_hashes(tmp_path: Path):
    _write(tmp_path, "prompts")
    p = load_policy("prompts", policy_dir=tmp_path)
    assert p.question == "Is it bad?" and p.malicious == "bad things" and p.benign == "good things"
    assert len(p.hash) == 12 and p.ref == f"shared@{p.hash}"


def test_hash_changes_with_text_and_examples(tmp_path: Path):
    _write(tmp_path, "prompts")
    a = load_policy("prompts", policy_dir=tmp_path)
    b = load_policy("prompts", examples=[("x", 1)], policy_dir=tmp_path)
    assert a.hash != b.hash
    _write(tmp_path, "prompts", "tuned")
    (tmp_path / "prompts" / "tuned.yaml").write_text(
        yaml.safe_dump({"question": "Is it evil?", "malicious": "bad", "benign": "good"}))
    c = load_policy("prompts", name="tuned", policy_dir=tmp_path)
    assert c.hash != a.hash and c.ref.startswith("tuned@")


def test_pick_examples_is_balanced_and_deterministic():
    items = [Item(id=f"i{i}", track="prompts", text=f"t{i}", label=i % 2) for i in range(40)]
    ex1 = pick_examples(items, n=8, seed=0)
    ex2 = pick_examples(items, n=8, seed=0)
    assert ex1 == ex2 and len(ex1) == 8
    assert sum(l for _, l in ex1) == 4


def test_real_policy_files_load():
    for track in ("prompts", "skills"):
        p = load_policy(track)
        assert p.question and p.malicious and p.benign
