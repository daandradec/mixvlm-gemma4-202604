# start_server.py
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_VLM_PORT"] = "8766"
MODEL_PATH = "/Users/daandradec/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef"

sys.argv = [
    "mlx_vlm.server",
    "--model", MODEL_PATH,
    "--kv-bits", "3.5",
    "--kv-quant-scheme", "turboquant",
    "--host", "0.0.0.0",
    "--port", "8766",
    "--max-kv-size", "32768",
]

from mlx_vlm.server import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8766)
