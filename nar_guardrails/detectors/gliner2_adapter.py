from __future__ import annotations

from nar_guardrails.detectors.base import Detector, Prediction, hf_revision, register
from nar_guardrails.detectors.policy import Policy
from nar_guardrails.device import pick_device

EMPTY_TEXT = "(empty)"


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
    """GLiNER2 classification detector.

    The GLiNER2.5-Decide checkpoint sets no `max_len`, so without a cap a 100k-character skill goes
    into DeBERTa-large whole and attention memory explodes (observed: 26 GB resident, swap exhausted).
    `max_length` (tokens of the model's own tokenizer) bounds every forward pass; `window: true` scores
    `max_length`-token windows and takes the max, like the encoder-classifier adapter, otherwise the
    head is kept and the verdict is flagged `truncated`.
    """

    @staticmethod
    def build_schema(mode: str, policy: Policy) -> tuple[dict, str, str]:
        """Returns (schema, task_name, positive_label).

        gliner2 accepts {task: {label: description}} for described labels.
        """
        if mode == "gliguard":
            return {"prompt_safety": ["safe", "unsafe"]}, "prompt_safety", "unsafe"
        if mode != "decide":
            raise ValueError(f"unknown gliner2 mode {mode!r}; expected 'decide' or 'gliguard'")
        schema = {"input_safety": {"malicious": policy.malicious, "benign": policy.benign}}
        return schema, "input_safety", "malicious"

    def __init__(self, cfg: dict, policy: Policy):
        super().__init__(cfg, policy)
        from gliner2 import GLiNER2  # pip install "gliner2[local]"

        self.mode = cfg.get("mode", "decide")
        self.max_length = int(cfg["max_length"]) if cfg.get("max_length") else None
        self.window = bool(cfg.get("window", False))
        if self.window and not self.max_length:
            raise ValueError("gliner2: window=true needs max_length")
        self.model = GLiNER2.from_pretrained(cfg["model_id"])
        try:
            self.model.to(pick_device(cfg.get("device", "auto")))
        except Exception:
            self.model.to("cpu")
        self.tok = None
        if self.max_length:
            from transformers import AutoTokenizer  # the GLiNER2 snapshot ships its encoder tokenizer

            self.tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg.get("revision"))
        self.schema, self.task, self.positive = self.build_schema(self.mode, policy)
        self.revision = hf_revision(cfg["model_id"], cfg.get("revision"))

    def _classify(self, texts: list[str]) -> tuple[list[tuple[str, float | None]], list]:
        bs = int(self.cfg.get("batch_size", 8))
        if len(texts) > 1:
            results = self.model.batch_classify_text(texts, self.schema, include_confidence=True, batch_size=bs)
        else:
            results = [self.model.classify_text(t, self.schema, include_confidence=True) for t in texts]
        return [extract_label_confidence(r, self.task) for r in results], results

    def _probability(self, label: str, conf: float | None) -> float:
        is_pos = label.lower() == self.positive
        if conf is None:
            return 1.0 if is_pos else 0.0
        return conf if is_pos else 1.0 - conf

    def _windows(self, text: str) -> list[str]:
        """`max_length`-token windows of `text` (no overlap); the head window alone is the truncation."""
        ids = self.tok.encode(text, add_special_tokens=False)
        step = max(1, self.max_length - 2)
        chunks = [ids[i:i + step] for i in range(0, len(ids), step)] or [[]]
        return [self.tok.decode(c) or EMPTY_TEXT for c in chunks]

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        texts = [t if t.strip() else EMPTY_TEXT for t in texts]
        if self.window:  # windows of one text form one forward batch; the text scores as its worst window
            out: list[Prediction] = []
            for text in texts:
                windows = self._windows(text)
                scored, _ = self._classify(windows)
                probs = [self._probability(label, conf) for label, conf in scored]
                p = max(probs)
                out.append(Prediction(p, int(p >= 0.5), "ok", None, {"windows": len(windows), "window_probs": probs}))
            return out
        outcomes = ["ok"] * len(texts)
        if self.max_length:
            n_tokens = [len(self.tok.encode(t, add_special_tokens=False)) for t in texts]
            texts = [self._windows(t)[0] if n > self.max_length else t for t, n in zip(texts, n_tokens)]
            outcomes = ["truncated" if n > self.max_length else "ok" for n in n_tokens]
        scored, raw = self._classify(texts)
        return [Prediction(self._probability(label, conf), int(label.lower() == self.positive), outcome, None, r)
                for (label, conf), outcome, r in zip(scored, outcomes, raw)]
