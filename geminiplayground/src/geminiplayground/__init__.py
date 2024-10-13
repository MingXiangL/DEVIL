import logging

from rich.logging import RichHandler

from geminiplayground.core import GeminiClient

__all__ = ["GeminiClient"]

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False)],
)
