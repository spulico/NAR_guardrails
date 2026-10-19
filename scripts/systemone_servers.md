# Running the optional SystemOne-style detectors locally

Both servers expose `POST /v1/systemone`; the `openjev` adapter talks to either. The runner skips a detector automatically when its health check fails.

## Open-Jev-2B (`openjev-2b`, port 8791)

```bash
git clone https://github.com/Zefan-Cai/Open-Jev.git external/Open-Jev
cd external/Open-Jev
uv venv --python 3.12 .venv && source .venv/bin/activate
python -m pip install -e '.[train]'
hf download ZefanCai/Open-Jev-2B --revision 0c7aa498b1627be8da4acf34c863ff0ee0a92785 --local-dir models/Open-Jev-2B
python -m jev.server --checkpoint models/Open-Jev-2B/package/checkpoint --device mps --max-length 4096 --batch-size 1 --no-prefix-cache
```

The server listens on port 8791 and exposes `/health` and `POST /v1/systemone`. If `--device mps` fails, use `--device cpu` (slow) or skip the detector; the runner skips it automatically when the health check fails.

## basal-2.0-mini (`basal-2.0-mini`, port 8000)

The basal engine pins its own torch build, so it must live in its own venv, separate from this project's `.venv` and from the Open-Jev venv.

Apple Silicon (MLX), per the rkinas/basal README:

```bash
uv venv --python 3.12 --allow-existing ~/basal-mlx && source ~/basal-mlx/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
basal-serve --model Remek/basal-2.0-mini --mode mlx --port 8000
```

GGUF / llama.cpp alternative: `basal-serve --model <gguf model> --mode gguf --port 8000`. The README gives no install commands for it; see its `docs/INSTALL.md#ollama-and-llamacpp-gguf`.

The README documents no health endpoint (only `POST /v1/systemone` and `/mcp`), so `configs/models.yaml` keeps the default `/health`. If your basal build uses another path, set `health_url` on the `basal-2.0-mini` entry. Verify with `curl http://127.0.0.1:8000/health`.

Response shape: `{"answers": {"malicious": {"type": "noul", "noul": p_yes, "probabilities": {...}, "confidence": c}}, "usage": {...}}`.
