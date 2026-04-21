from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mlx_vlm Codex Proxy")

BACKEND_BASE_URL = os.getenv("MLX_PROXY_BACKEND_URL", "http://127.0.0.1:8765").rstrip(
    "/"
)
BACKEND_RESPONSE_PATHS = ("/responses", "/v1/responses")
TIMEOUT_S = float(os.getenv("MLX_PROXY_TIMEOUT_S", "300"))
FORWARD_MODEL = os.getenv("MLX_PROXY_FORWARD_MODEL", "0").lower() in ("1", "true", "yes")
BACKEND_MODEL = os.getenv("MLX_PROXY_BACKEND_MODEL", "").strip()


def _normalize_role(role: Any) -> str:
    value = str(role or "user").lower()
    if value == "developer":
        return "system"
    if value in ("user", "assistant", "system"):
        return value
    return "user"


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("text", "input_text", "output_text"):
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif item_type == "message":
            msg_content = item.get("content")
            msg_text = _text_from_content(msg_content)
            if msg_text:
                parts.append(msg_text)

    return "\n".join(parts).strip()


def _normalize_input(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]

    if not isinstance(raw, list):
        return [{"role": "user", "content": str(raw) if raw is not None else ""}]

    messages: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        if item.get("type") == "message":
            role = _normalize_role(item.get("role", "user"))
            content = _text_from_content(item.get("content"))
            messages.append({"role": role, "content": content})
            continue

        if "role" in item:
            role = _normalize_role(item.get("role", "user"))
            content = _text_from_content(item.get("content"))
            if not content and isinstance(item.get("content"), str):
                content = item["content"]
            messages.append({"role": role, "content": content})
            continue

        if item.get("type") in ("text", "input_text"):
            text = item.get("text")
            if isinstance(text, str):
                messages.append({"role": "user", "content": text})

    if not messages:
        return [{"role": "user", "content": ""}]
    return messages


def _build_backend_payload(body: dict[str, Any], force_stream: bool | None) -> dict[str, Any]:
    stream = bool(body.get("stream", False)) if force_stream is None else force_stream
    raw_input = body.get("input", body.get("messages", ""))
    model_value = BACKEND_MODEL
    if not model_value and FORWARD_MODEL:
        model_value = str(body.get("model", "")).strip()
    if not model_value:
        model_value = "local-model"

    payload: dict[str, Any] = {
        "model": model_value,
        "input": _normalize_input(raw_input),
        "stream": stream,
    }

    passthrough_keys = (
        "max_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    )
    for key in passthrough_keys:
        if key in body:
            payload[key] = body[key]

    return payload


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(
        usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    )
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _extract_output_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str):
        return direct

    for key in ("response", "text", "content"):
        value = data.get(key)
        if isinstance(value, str):
            return value

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                content_text = _text_from_content(content)
                if content_text:
                    return content_text

    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                return content
            text = _text_from_content(content)
            if text:
                return text

    return ""


def _to_responses_format(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("object") == "response":
        return data

    output_text = _extract_output_text(data)
    usage = _extract_usage(data)
    created_at = int(time.time())
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": data.get("model", "mlx_vlm"),
        "error": None,
        "output_text": output_text,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": output_text, "annotations": []}],
            }
        ],
        "usage": usage,
    }


def _to_chat_completions_format(data: dict[str, Any], model: str | None) -> dict[str, Any]:
    output_text = _extract_output_text(data)
    usage = _extract_usage(data)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or data.get("model", "mlx_vlm"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    }


async def _post_backend_json(payload: dict[str, Any]) -> dict[str, Any]:
    timeout = httpx.Timeout(TIMEOUT_S, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        last_error: str | None = None
        for path in BACKEND_RESPONSE_PATHS:
            url = f"{BACKEND_BASE_URL}{path}"
            response = await client.post(url, json=payload)
            if response.status_code == 404:
                last_error = f"{url} -> 404 {response.text}"
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"{url} -> {response.status_code} {response.text}")
            return response.json()
    raise RuntimeError(last_error or "No backend /responses endpoint found.")


async def _proxy_backend_stream(payload: dict[str, Any]):
    timeout = httpx.Timeout(TIMEOUT_S, connect=20.0, read=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in BACKEND_RESPONSE_PATHS:
            url = f"{BACKEND_BASE_URL}{path}"
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code == 404:
                    continue
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "ignore")
                    err = {"type": "error", "message": f"{url} -> {response.status_code} {body}"}
                    yield f"event: error\ndata: {json.dumps(err)}\n\n".encode("utf-8")
                    return

                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
                return

    err = {"type": "error", "message": "No backend /responses endpoint found."}
    yield f"event: error\ndata: {json.dumps(err)}\n\n".encode("utf-8")


@app.post("/v1/responses")
@app.post("/responses")
async def responses(request: Request):
    body = await request.json()
    payload = _build_backend_payload(body, force_stream=None)

    if payload.get("stream"):
        return StreamingResponse(
            _proxy_backend_stream(payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        data = await _post_backend_json(payload)
        return JSONResponse(_to_responses_format(data))
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "backend_error", "message": str(exc)}},
        )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    payload = _build_backend_payload(body, force_stream=False)

    try:
        data = await _post_backend_json(payload)
        return JSONResponse(_to_chat_completions_format(data, body.get("model")))
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "backend_error", "message": str(exc)}},
        )


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "mlx_vlm",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND_BASE_URL}


if __name__ == "__main__":
    host = os.getenv("MLX_PROXY_HOST", "0.0.0.0")
    port = int(os.getenv("MLX_PROXY_PORT", "8766"))
    uvicorn.run(app, host=host, port=port)
