from __future__ import annotations

from nar_guardrails.detectors.base import Detector, Prediction, hf_revision, register
from nar_guardrails.device import pick_device

LEVEL_SCORE = {"safe": 0.0, "controversial": 0.5, "unsafe": 1.0}


def score_from_moderation(m: dict) -> tuple[float, int]:
    label = str(m.get("label", "Safe")).lower()
    probs = m.get("probs")
    if probs:
        norm = {str(k).lower(): float(v) for k, v in probs.items()}
        p = 1.0 - norm.get("safe", 0.0)
    else:
        p = LEVEL_SCORE.get(label, 0.5)
    return p, int(label == "unsafe")


def _ensure_default_rope() -> None:
    """transformers 5 dropped ROPE_INIT_FUNCTIONS["default"], which the model's remote code
    still looks up. Restore the standard (unscaled) RoPE inverse frequencies."""
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def compute_default(config, device=None, seq_len=None, layer_type=None):
        params = getattr(config, "rope_parameters", None) or {}
        base = params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(dim * params.get("partial_rotary_factor", 1.0))
        idx = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
        return 1.0 / (base ** (idx / dim)), 1.0

    ROPE_INIT_FUNCTIONS["default"] = compute_default


def _patch_remote_masks(model) -> None:
    """The remote code calls create_causal_mask(input_embeds=..., cache_position=...); transformers 5
    renamed the first and dropped the second. Adapt the module-level names the remote code uses."""
    import sys

    mod = sys.modules[type(model).__module__]
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        fn = getattr(mod, name, None)
        if fn is None or getattr(fn, "_nar_patched", False):
            continue

        def wrapped(*args, _fn=fn, **kw):
            if "input_embeds" in kw:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            kw.pop("cache_position", None)
            return _fn(*args, **kw)

        wrapped._nar_patched = True
        setattr(mod, name, wrapped)


@register("qwen3guard")
class Qwen3GuardDetector(Detector):
    def __init__(self, cfg: dict, policy):
        super().__init__(cfg, policy)
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.device = pick_device(cfg.get("device", "auto"))
        _ensure_default_rope()
        rev = cfg.get("revision")
        # trust_remote_code executes Hub code: pin `revision` in configs/models.yaml.
        self.tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True,
                                                 revision=rev)
        # The remote code reads config.pad_token_id; transformers 5 no longer defaults it. This only
        # guarantees the attribute exists (value None if unset).
        config = AutoConfig.from_pretrained(cfg["model_id"], trust_remote_code=True, revision=rev)
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        self.model = AutoModel.from_pretrained(cfg["model_id"], trust_remote_code=True,
                                               revision=rev, config=config,
                                               torch_dtype=torch.float32).to(self.device).eval()
        _patch_remote_masks(self.model)
        self.revision = hf_revision(cfg["model_id"], rev, config)

    def _moderate(self, text: str) -> dict:
        # Model card: tokenize the chat, feed ids up to the end of the user turn
        # (<|im_end|>) to stream_moderate_from_ids(role="user", stream_state=None).
        import torch

        chat = [{"role": "user", "content": text}]
        prompt = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False,
                                              enable_thinking=False)
        ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        end = (ids == im_end).nonzero()[0].item()
        state = None
        try:
            with torch.no_grad():
                result, state = self.model.stream_moderate_from_ids(
                    ids[: end + 1].to(self.device), role="user", stream_state=None)
            risk = str(result["risk_level"][-1])
            category = result.get("category")
            category = category[-1] if isinstance(category, (list, tuple)) and category else category
            rp = result.get("risk_prob")
            rp = float(rp[-1] if isinstance(rp, (list, tuple)) else rp) if rp is not None else None
        finally:
            if state is not None:
                self.model.close_stream(state)
        # risk_prob is the probability of the predicted class only; derive a binary Safe/Unsafe split.
        probs = None
        if rp is not None and risk.lower() == "safe":
            probs = {"Safe": rp, "Unsafe": 1.0 - rp}
        elif rp is not None and risk.lower() == "unsafe":
            probs = {"Safe": 1.0 - rp, "Unsafe": rp}
        return {"label": risk, "probs": probs, "risk_prob": rp,
                "category": None if category is None else str(category)}

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        out = []
        for t in texts:
            m = self._moderate(t if t.strip() else "(empty)")
            p, label = score_from_moderation(m)
            out.append(Prediction(p, label, "ok", None, m))
        return out
