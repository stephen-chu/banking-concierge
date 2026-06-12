"""FastAPI custom routes for the LangGraph agent server.

Mounted at the deployment root via `langgraph.json`'s `http.app` key, so
this app extends the agent server (which already exposes `/threads`,
`/runs`, `/assistants`, etc.) with our own routes:

- `GET  /health`               - simple liveness probe
- `GET  /`                     - redirect to `/concierge/`
- `GET  /concierge/`           - the built React chat UI (Vite dist/)
- `POST /concierge-api/feedback` - record thumbs up/down on a run in LangSmith

If the frontend bundle hasn't been built yet, `/concierge/*` returns a
503 with a hint to run `npm install && npm run build` in `frontend/`.

The feedback route keeps `LANGSMITH_API_KEY` server-side: the browser
posts only `{run_id, score, comment}` and this app calls the LangSmith
SDK on its behalf, so the key is never shipped to the client.

NOTE: do NOT add `from __future__ import annotations` here. The custom app's
request models (e.g. FeedbackRequest) must resolve eagerly so the LangGraph
server can build the OpenAPI spec at startup; lazy string annotations raise
PydanticUserError("...is not fully defined").
"""

import hashlib
import json
import os
import pathlib
import re
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langsmith import Client
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()


_RUN_CREATE_PATH = re.compile(
    r"^/(?:threads/[^/]+/runs|runs)(?:/(?:stream|wait|batch))?/?$"
)


def _rep_identifier(request: Request) -> str | None:
    """Derive a stable, non-secret id for the authenticated rep from the request."""
    api_key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if not api_key:
        return None
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"rep-{digest}"


class RunMetadataMiddleware(BaseHTTPMiddleware):
    """Inject `user_id` and `environment` into LangGraph run-creation metadata.

    LangSmith needs a non-empty `user_id` on root runs to power per-rep
    filtering on this multi-tenant app. We attach it at the HTTP boundary so
    every run created through the agent server picks it up regardless of which
    client initiated it.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and _RUN_CREATE_PATH.match(request.url.path):
            user_id = _rep_identifier(request)
            if user_id:
                body = await request.body()
                if body:
                    try:
                        payload = json.loads(body)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        config = payload.setdefault("config", {})
                        if isinstance(config, dict):
                            metadata = config.setdefault("metadata", {})
                            if isinstance(metadata, dict):
                                metadata.setdefault("user_id", user_id)
                                metadata.setdefault(
                                    "environment",
                                    os.getenv("CONCIERGE_ENV", "production"),
                                )
                                new_body = json.dumps(payload).encode("utf-8")
                                request._body = new_body  # noqa: SLF001

                                async def receive() -> dict:
                                    return {
                                        "type": "http.request",
                                        "body": new_body,
                                        "more_body": False,
                                    }

                                request._receive = receive  # noqa: SLF001
        return await call_next(request)


app.add_middleware(RunMetadataMiddleware)

_langsmith_client = Client()


class FeedbackRequest(BaseModel):
    run_id: uuid.UUID
    score: float = Field(ge=0, le=1)
    comment: str | None = None


@app.post("/concierge-api/feedback")
def submit_feedback(body: FeedbackRequest) -> dict[str, str]:
    """Attach user thumbs up/down to the LangSmith run that produced a reply.

    `score` is 1 for thumbs-up and 0 for thumbs-down. Passing `trace_id`
    lets the SDK background the write so the request returns immediately.
    """
    try:
        _langsmith_client.create_feedback(
            body.run_id,
            key="user_feedback",
            score=body.score,
            trace_id=body.run_id,
            comment=body.comment,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 502 to the client
        raise HTTPException(
            status_code=502, detail=f"Failed to record feedback: {exc}"
        ) from exc
    return {"status": "ok"}

FRONTEND_BUILD_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/concierge/")


@app.get("/concierge")
def concierge_redirect() -> RedirectResponse:
    return RedirectResponse(url="/concierge/")


if FRONTEND_BUILD_DIR.is_dir() and (FRONTEND_BUILD_DIR / "index.html").is_file():
    app.mount(
        "/concierge",
        StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True),
        name="frontend",
    )
else:

    @app.get("/concierge/{path:path}")
    def frontend_not_built(path: str = "") -> PlainTextResponse:
        del path
        return PlainTextResponse(
            "Frontend not built. "
            "From the project root, run:\n\n"
            "  npm --prefix frontend install\n"
            "  npm --prefix frontend run build\n",
            status_code=503,
        )
