MODEL_PATH="/Users/daandradec/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef"

source ./mlx-gemma4/bin/activate
HF_HUB_OFFLINE=1 MLX_VLM_PORT=8765 mlx_vlm.server --model "$MODEL_PATH" --kv-bits 3.5 --kv-quant-scheme turboquant --host 0.0.0.0 --port 8765 --max-kv-size 32768
#python start-server.py
