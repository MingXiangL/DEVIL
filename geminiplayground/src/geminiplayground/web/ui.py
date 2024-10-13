import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

ui = FastAPI(root_path="")

root_file_path = os.path.dirname(os.path.abspath(__file__))
static_folder_root = os.path.join(root_file_path, "static")

ui.mount("/", StaticFiles(directory=static_folder_root, html=True), name="static")
