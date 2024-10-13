import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .api import api
from .ui import ui
from .db.session_manager import sessionmanager
import logging

logger = logging.getLogger("rich")

from .db.models import *

BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = Path(BASE_DIR, "files")
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))

os.environ["FILES_DIR"] = FILES_DIR.as_posix()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Api life span
    :return:
    """
    logger.info("ui is starting")
    await sessionmanager.init()
    yield
    logger.info("ui is shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=FILES_DIR), name="files")


@app.get("/files", response_class=HTMLResponse)
def list_files(request: Request):
    """
    List files in the files directory
    """
    files = os.listdir(FILES_DIR)
    files_paths = sorted([f"{request.url._url}/{f}" for f in files])
    return templates.TemplateResponse(
        request=request, name="files.html", context={"files": files_paths}
    )


apps = {
    "/api": api,
    "/": ui,
}
for path, sub_app in apps.items():
    app.mount(path, sub_app)
