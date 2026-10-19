from __future__ import annotations

from nar_guardrails.detectors.base import Detector, Prediction, register
from nar_guardrails.device import pick_device


@register("hf_classifier")
class HFClassifier(Detector):
    """Binary sequence classifier from the Hub. p_malicious = softmax prob of `positive_label`."""

    def __init__(self, cfg: dict, policy):
        super().__init__(cfg, policy)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = pick_device(cfg.get("device", "auto"))
        self.tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg.get("revision"))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_id"], revision=cfg.get("revision")).to(self.device).eval()
        self.max_length = int(cfg.get("max_length", 512))
        self.window = bool(cfg.get("window", False))
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        wanted = str(cfg["positive_label"]).upper()
        matches = [i for i, l in id2label.items() if l.upper() == wanted]
        if not matches:
            raise ValueError(f"{cfg['model_id']}: positive_label {cfg['positive_label']!r} not in {id2label}")
        self.pos_idx = matches[0]
        self._torch = torch

    def _scores(self, texts: list[str]) -> list[float]:
        torch = self._torch
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, self.pos_idx]
        return [float(x) for x in probs.cpu().tolist()]

    def _windows(self, text: str) -> list[str]:
        ids = self.tok.encode(text, add_special_tokens=False)
        step = max(1, self.max_length - 2)
        chunks = [ids[i:i + step] for i in range(0, len(ids), step)] or [[]]
        return [self.tok.decode(c) for c in chunks]

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        texts = [t if t.strip() else "(empty)" for t in texts]
        if self.window:  # windows of one text form one forward batch
            out: list[Prediction] = []
            for text in texts:
                windows = self._windows(text)
                p = max(self._scores(windows))
                out.append(Prediction(p, int(p >= 0.5), "ok", None, {"windows": len(windows)}))
            return out
        n_tokens = [len(self.tok.encode(t, add_special_tokens=False)) + 2 for t in texts]
        scores = self._scores(texts) if texts else []  # one padded forward pass for the whole batch
        return [Prediction(p, int(p >= 0.5), "truncated" if n > self.max_length else "ok", None, {"n_tokens": n})
                for p, n in zip(scores, n_tokens)]
