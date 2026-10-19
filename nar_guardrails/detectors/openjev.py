from __future__ import annotations

import requests

from nar_guardrails.detectors.base import Detector, Prediction, register
from nar_guardrails.detectors.laya_adapter import noul_probability


class DetectorUnavailable(RuntimeError):
    """Raised when an optional detector's backend is not reachable."""


@register("openjev")
class OpenJevDetector(Detector):
    """Jev-style /v1/systemone client; serves Open-Jev-2B and basal-2.0-mini."""

    def __init__(self, cfg: dict, policy, session=None):
        super().__init__(cfg, policy)
        self.url = cfg["url"]
        self.timeout = float(cfg.get("timeout_s", 60))
        self.session = session or requests.Session()
        self.questions = {"malicious": {"type": "noul", "instructions": policy.question}}
        health = cfg.get("health_url") or self.url.rsplit("/v1/", 1)[0] + "/health"
        self.health_mode = None
        try:
            if self.session.get(health, timeout=5).status_code == 200:
                self.health_mode = "get"
        except Exception:
            pass
        if self.health_mode is None:
            try:
                r = self.session.post(self.url, json={"state": "ping", "questions": self.questions},
                                      timeout=min(self.timeout, 15))
                if r.status_code == 200:
                    self.health_mode = "probe"
            except Exception:
                pass
        if self.health_mode is None:
            raise DetectorUnavailable(
                f"SystemOne server not reachable (GET {health} and probe POST {self.url} failed); "
                "see scripts/systemone_servers.md")

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        out = []
        for t in texts:
            r = self.session.post(self.url, json={"state": t if t.strip() else "(empty)",
                                                  "questions": self.questions}, timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
            p = noul_probability(body["answers"]["malicious"])
            out.append(Prediction(p, int(p >= 0.5), "ok", body.get("usage"), body["answers"]))
        return out
