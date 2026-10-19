from pathlib import Path
from nar_guardrails.bench.smoke import smoke


def test_smoke_runs_fake_on_both_tracks(tmp_path: Path):
    df = smoke({"fake": {"adapter": "fake", "tracks": ["prompts", "skills"]}}, tmp_path, n=6, detectors=["fake"])
    assert set(df["track"]) == {"prompts", "skills"} and (df["status"] == "ok").all()
    assert (df["n_ok"] == 6).all()
    p = df[df["track"] == "prompts"].iloc[0]
    assert p["mean_p_malicious"] > p["mean_p_benign"]   # skills look-alikes legitimately contain curl


def test_cli_smoke_exits_zero(monkeypatch, tmp_path, capsys):
    from nar_guardrails.cli import main
    monkeypatch.setattr("nar_guardrails.bench.smoke.load_models_cfg", lambda: {"fake": {"adapter": "fake", "tracks": ["prompts", "skills"]}})
    assert main(["smoke", "--results-dir", str(tmp_path), "--detectors", "fake"]) == 0
    assert "fake" in capsys.readouterr().out
