from nar_guardrails.detectors.base import Detector, Prediction, register

KEYWORDS = ("ignore previous", "ignore all previous", "system prompt", "disregard", "exfiltrat",
            "curl ", "~/.ssh", "developer mode", "do not tell the user", "base64 -d")


@register("fake")
class FakeDetector(Detector):
    """Keyword scorer used by unit tests and `nar smoke`. Not a real guardrail."""

    def predict_texts(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            hits = sum(k in low for k in KEYWORDS)
            p = min(0.95, 0.1 + 0.4 * hits)
            out.append(Prediction(p, int(p >= 0.5), "ok", None, {"hits": hits}))
        return out
