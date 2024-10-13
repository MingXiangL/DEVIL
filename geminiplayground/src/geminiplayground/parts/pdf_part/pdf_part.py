import logging
import typing

from PyPDF2 import PdfReader

from geminiplayground.core import GeminiClient
from .. import MultimodalPart
from geminiplayground.utils import FileUtils
from pathlib import Path

logger = logging.getLogger("rich")


class PdfFile(MultimodalPart):
    """
    PDF file part implementation
    """

    def __init__(self, pdf_path: typing.Union[str, Path], **kwargs):
        self.path = pdf_path
        self.filename = FileUtils.get_file_name_from_path(pdf_path)
        self.gemini_client = kwargs.get("gemini_client", GeminiClient())

    def __get_pdf_parts(self) -> typing.List[str]:
        """
        Get the content parts for the pdf
        :return: list of text parts
        """
        text_parts = []
        with FileUtils.solve_file_path(self.path) as pdf_path:
            with open(pdf_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    text_parts.append(page.extract_text())
            return text_parts

    def content_parts(self):
        """
        Get the content parts for the pdf
        :return:
        """
        return self.__get_pdf_parts()
