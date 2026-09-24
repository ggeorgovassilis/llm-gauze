"""Bandaid gateway entrypoint.

Exposes an OpenAI-compatible API and forwards everything to the configured
local LLM provider, recording every exchange along the way.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("bandaid")

record_path = Path(settings.data_dir) / settings.record_file
recorder = Recorder(record_path)
proxy = Proxy(recorder)

app = FastAPI(title="Bandaid Gateway")

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {"status": "ok", "upstream": settings.llm_base_url}
    )


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def catch_all(request: Request, path: str) -> Response:
    return await proxy.forward(request, path)


app.include_router(router)

logger.info("Bandaid gateway ready, forwarding to %s", settings.llm_base_url)
