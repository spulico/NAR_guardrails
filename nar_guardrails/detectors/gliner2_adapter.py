from __future__ import annotations

from nar_guardrails.detectors.base import Detector, Prediction, hf_revision, register
from nar_guardrails.detectors.policy import Policy
from nar_guardrails.device import pick_device


def extract_label_confidence(result, task: str) -> tuple[str, float | None]:
    """Normalise the shapes gliner2 returns for classify_text(..., include_confidence=True)."""
    val = result[task] if isinstance(result, dict) and task in result else result
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, tuple) and len(val) == 2:
        return str(val[0]), float(val[1])
    if isinstance(val, str):
        return val, None
    if isinstance(val, dict):
        label = val.get("label") or val.get("class") or val.get("value")
        conf = val.get("confidence", val.get("score", val.get("probability")))
        return str(label), (float(conf) if conf is not None else None)
    raise ValueError(f"unrecognised gliner2 result shape for task {task!r}: {result!r}")


@register("gliner2")
class GLiNER2Detector(Detector):
    @staticmethod
    def build_schema(mode: str, policy: Policy) -> tuple[dict, str, str]:
        """Returns (schema, task_name, positive_label).

        gliner2 accepts {task: {label: description}} for described labels.
        """
        if mode == "gliguard":
            return {"prompt_safety": ["safe", "unsafe"]}, "prompt_safety", "unsafe"
        schema = {"input_safety": {"malicious": policy.malicious, "benign": policy.benign}}
        return schema, "input_safety", "malicious"

    def __init__(self, cfg: dict, policy: Policy):
        super().__init__(cfg, policy)
        from gliner2 import GLiNER2  # pip install "gliner2[local]"

        self.mode = cfg.get("mode", "decide")
        self.model = GLiNER2.from_pretrained(cfg["model_id"])
        try:
            self.model.to(pick_device(cfg.get("device", "auto")))
        except Exception:
            self.model.to("cpu")
        self.schema, self.task, self.positive = self.build_schema(self.mode, policy)
        self.revision = hf_revision(cfg["model_id"], cfg.get("revision"))

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        texts = [t if t.strip() else "(empty)" for t in texts]
        bs = int(self.cfg.get("batch_size", 8))
        if len(texts) > 1:
            results = self.model.batch_classify_text(texts, self.schema, include_confidence=True, batch_size=bs)
        else:
            results = [self.model.classify_text(t, self.schema, include_confidence=True) for t in texts]
        out = []
        for r in results:
            label, conf = extract_label_confidence(r, self.task)
            is_pos = label.lower() == self.positive
            if conf is None:
                p = 1.0 if is_pos else 0.0
            else:
                p = conf if is_pos else 1.0 - conf
            out.append(Prediction(p, int(is_pos), "ok", None, r))
        return out
