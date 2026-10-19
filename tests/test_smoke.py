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


def _fake_cfg():
    return {"fake": {"adapter": "fake", "tracks": ["prompts", "skills"]}}


def test_smoke_does_not_touch_shared_cache_or_latest(tmp_path: Path):
    df = smoke(_fake_cfg(), tmp_path, n=6, detectors=["fake"])
    assert not (tmp_path / "results" / "cache").exists() and not (tmp_path / "cache").exists()
    assert not (tmp_path / "LATEST").exists()
    run_dirs = [d for d in tmp_path.iterdir() if d.name.startswith("smoke-")]
    assert len(run_dirs) == 2 and all((d / "cache").is_dir() for d in run_dirs)
    assert "separates" in df.columns


def test_smoke_accepts_all_and_whitespace(tmp_path: Path):
    df = smoke(_fake_cfg(), tmp_path, n=4, detectors=["all", " fake "])
    assert set(df["detector"]) == {"fake"}


def _register_broken():
    from nar_guardrails.detectors.base import Detector, Prediction, register

    @register("always_error_test")
    class Broken(Detector):
        def predict_texts(self, texts):
            return [Prediction(None, None, "error", None, "boom: backend down") for _ in texts]


def test_smoke_all_error_detector_is_failed_and_cli_exits_one(monkeypatch, tmp_path, capsys):
    from nar_guardrails.cli import main
    _register_broken()
    cfg = {"broken": {"adapter": "always_error_test", "tracks": ["prompts", "skills"]}}
    df = smoke(cfg, tmp_path, n=6, detectors=["broken"])
    assert (df["status"] == "failed").all() and (df["n_ok"] == 0).all()
    assert df["reason"].iloc[0] == "all verdicts are errors: boom: backend down"
    monkeypatch.setattr("nar_guardrails.bench.smoke.load_models_cfg", lambda: cfg)
    assert main(["smoke", "--results-dir", str(tmp_path / "r2"), "--detectors", "broken", "--n", "6"]) == 1


def test_smoke_named_detector_runs_only_on_its_tracks(tmp_path: Path):
    import pytest
    cfg = {"fake": {"adapter": "fake", "tracks": ["prompts", "skills"]}, "ponly": {"adapter": "fake", "tracks": ["prompts"]}}
    df = smoke(cfg, tmp_path, n=4, detectors=["ponly"])
    assert list(zip(df["track"], df["detector"])) == [("prompts", "ponly")]
    with pytest.raises(KeyError):
        smoke(cfg, tmp_path, n=4, detectors=["nope"])
    with pytest.raises(ValueError):
        smoke(cfg, tmp_path, n=4, detectors=[" "])
