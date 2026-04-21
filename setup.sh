#!/bin/bash

set -euo pipefail

# Activar entorno virtual
source ./mlx-gemma4/bin/activate

MODEL_PATH="${MODEL_PATH:-/Users/daandradec/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef}"
BACKEND_PORT="${BACKEND_PORT:-8766}"
PROXY_PORT="${PROXY_PORT:-8765}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

echo "Starting mlx_vlm.server on ${BACKEND_PORT}..."
HF_HUB_OFFLINE=1 MLX_VLM_PORT="${BACKEND_PORT}" mlx_vlm.server \
  --model "${MODEL_PATH}" \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant \
  --host 0.0.0.0 \
  --port "${BACKEND_PORT}" \
  --max-kv-size 32768 2>&1 | tee server.log &

SERVER_PID=$!
echo "mlx_vlm.server PID: ${SERVER_PID}"

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for backend at ${BACKEND_URL}..."
for i in {1..90}; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Error: mlx_vlm.server stopped during startup. Check server.log"
    exit 1
  fi

  http_code="$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/health" || true)"
  if [[ "${http_code}" != "000" ]]; then
    echo "Backend is reachable (HTTP ${http_code})."
    break
  fi

  sleep 2
done

if [[ "${http_code:-000}" == "000" ]]; then
  echo "Error: backend did not become reachable on ${BACKEND_URL}"
  exit 1
fi

echo "Starting proxy on ${PROXY_PORT}..."
MLX_PROXY_BACKEND_URL="${BACKEND_URL}" \
MLX_PROXY_BACKEND_MODEL="${MODEL_PATH}" \
MLX_PROXY_PORT="${PROXY_PORT}" \
python proxy.py

# no funciono porque siempre sale error de Model type gemma4 not supported. porque la libreria no tiene gemma, el servidor se puede correr pero el api de response/v1 no funcionara nunca con Codex
# es mejor usar otra libreria que si este al dia (esta solo esta hasta gemma3) y que soporte hacer 'turboquant'