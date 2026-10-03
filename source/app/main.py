"""llm-gauze gateway entrypoint.

Exposes an OpenAI-compatible API and forwards everything to the configured
local LLM provider, recording every exchange along the way.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder
from app.telemetry import telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("llm_gauze")

record_path = Path(settings.data_dir) / settings.record_file
recorder = Recorder(record_path)
proxy = Proxy(recorder)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Tear down the upstream client's connection pool on shutdown.

    Without this hook ``proxy.client`` and its pool are abandoned at process
    exit; ``aclose()`` drains and closes them cleanly.
    """
    yield
    await proxy.aclose()


app = FastAPI(title="llm-gauze Gateway", lifespan=lifespan)

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "upstream": settings.llm_base_url})


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return JSONResponse(telemetry.snapshot())
    return Response(
        telemetry.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def catch_all(request: Request, path: str) -> Response:
    return await proxy.forward(request, path)


app.include_router(router)

logger.info("llm-gauze gateway ready, forwarding to %s", settings.llm_base_url)
