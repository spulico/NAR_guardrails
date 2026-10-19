from __future__ import annotations

from nar_guardrails.detectors.base import Detector, Prediction, hf_revision, register
from nar_guardrails.detectors.policy import Policy
from nar_guardrails.device import pick_device


def noul_probability(answer: dict) -> float:
    """Laya/Jev-style noul answers: return P(yes). Accepts the shapes seen in the Laya and Open-Jev docs."""
    for key in ("probability", "p_true", "noul", "p_yes", "yes"):
        if key in answer and isinstance(answer[key], (int, float)):
            return float(answer[key])
    if "value" in answer and "confidence" in answer:
        c = float(answer["confidence"])
        return c if bool(answer["value"]) else 1.0 - c
    raise KeyError(f"cannot find a noul probability in answer keys {sorted(answer)}")


@register("laya")
class LayaDetector(Detector):
    @staticmethod
    def build_questions(policy: Policy) -> dict:
        return {"malicious": {"type": "noul", "instructions": policy.question}}

    def __init__(self, cfg: dict, policy: Policy):
        super().__init__(cfg, policy)
        import laya  # pip install laya

        self.agent = laya.load(cfg["model_id"], device=pick_device(cfg.get("device", "auto")))
        if cfg.get("max_length"):
            # laya.load has no max_len kwarg; Agent reads it from its cfg dict
            self.agent.cfg["max_len"] = int(cfg["max_length"])
        self.questions = self.build_questions(policy)
        # laya's Agent records the snapshot commit it loaded (it honours LAYA_REVISION); fall back to the cache
        self.revision = getattr(self.agent, "revision", None) or hf_revision(cfg["model_id"], cfg.get("revision"))

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        states = [t if t.strip() else "(empty)" for t in texts]
        if len(states) == 1:
            results = [self.agent.predict(states[0], self.questions)]
        else:
            results = self.agent.predict_batch(states, self.questions,
                                               batch_size=int(self.cfg.get("batch_size", 8)))
        out = []
        for r in results:
            p = noul_probability(r["answers"]["malicious"])
            outcome = "truncated" if (r.get("usage") or {}).get("truncated") else "ok"  # input cut at max_len
            out.append(Prediction(p, int(p >= 0.5), outcome, r.get("usage"), r["answers"]))
        return out
