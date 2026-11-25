"""Serve the comparison UI and proxy the vLLM API underneath it.

    ./.venv-eval/bin/python ui/server.py            # port 8080 -> vLLM on 8000

Why a proxy rather than calling vLLM directly from the page: served this way,
the page and the API share one origin, so there is no CORS preflight, no mixed
content over the HTTPS port-forward, and no need to expose the raw inference
endpoint separately. The page fetches relative URLs ("/v1/chat/completions").

Streaming is passed straight through, so tokens appear as they are generated.
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

HERE = os.path.dirname(os.path.abspath(__file__))
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
PORT = int(os.environ.get("UI_PORT", "8080"))

app = FastAPI(title="Narrow finetune probe")
# Generous read timeout: a 600-token completion on a 32B model takes a while.
client = httpx.AsyncClient(
    base_url=VLLM_URL,
    timeout=httpx.Timeout(900.0, connect=10.0),
)


@app.get("/")
async def index():
    return FileResponse(
        os.path.join(HERE, "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/healthz")
async def healthz():
    try:
        r = await client.get("/v1/models", timeout=10.0)
        return JSONResponse({"ui": "ok", "vllm": r.status_code,
                             "models": [m["id"] for m in r.json().get("data", [])]})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ui": "ok", "vllm": "unreachable",
                             "error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def proxy(path: str, request: Request):
    body = await request.body()
    # Ask for identity so the raw byte stream we forward is not compressed.
    headers = {"accept-encoding": "identity"}
    for k in ("content-type", "accept"):
        if k in request.headers:
            headers[k] = request.headers[k]

    upstream = client.build_request(
        request.method, f"/v1/{path}", content=body, headers=headers,
        params=dict(request.query_params),
    )
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.HTTPError as e:
        return JSONResponse(
            {"error": {"message": f"cannot reach vLLM at {VLLM_URL}: {e}"}},
            status_code=502,
        )

    async def body_iter():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    passthrough = {
        k: v for k, v in resp.headers.items()
        if k.lower() in ("content-type", "cache-control")
    }
    passthrough.setdefault("Cache-Control", "no-store")
    # Tell any intermediate proxy not to buffer the SSE stream.
    passthrough["X-Accel-Buffering"] = "no"
    return StreamingResponse(body_iter(), status_code=resp.status_code,
                             headers=passthrough)


if __name__ == "__main__":
    import uvicorn

    print(f"[ui] serving {HERE}/index.html on 0.0.0.0:{PORT}")
    print(f"[ui] proxying /v1/* -> {VLLM_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
