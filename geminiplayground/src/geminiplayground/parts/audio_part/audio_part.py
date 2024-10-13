import logging
import typing

from geminiplayground.catching import cache
from geminiplayground.core import GeminiClient
from .. import MultimodalPart
from geminiplayground.utils import FileUtils, LibUtils
from pathlib import Path

logger = logging.getLogger("rich")


class AudioFile(MultimodalPart):
    """
    Audio file part implementation
    """

    def __init__(self, audio_path: typing.Union[str, Path], **kwargs):
        self.path = audio_path
        self.filename = FileUtils.get_file_name_from_path(audio_path)
        self.gemini_client = kwargs.get("gemini_client", GeminiClient())

    def upload(self):
        """
        Upload the audio to Gemini
        :return:
        """
        # upload the file
        with FileUtils.solve_file_path(self.path) as audio_file:
            uploaded_file = self.gemini_client.upload_file(audio_file)
            return uploaded_file

    @property
    def file(self):
        """
        Get the files
        :return:
        """
        if cache.get(self.filename):
            logger.info(f"Getting audio file {self.filename} from cache")
            cached_file = cache.get(self.filename)
            return cached_file

        logger.info(f"Uploading audio file {self.filename}")
        uploaded_file = self.upload()
        delta_t = LibUtils.get_uploaded_file_exp_date_delta_t(uploaded_file)
        cache.set(self.filename, uploaded_file, expire=delta_t)
        return uploaded_file

    def force_upload(self):
        """
        Force the upload of the audio
        :return:
        """
        self.delete()
        self.upload()

    def delete(self):
        """
        Delete the image from Gemini
        :return:
        """
        if cache.get(self.filename):
            cached_file = cache.get(self.filename)
            self.gemini_client.delete_file(cached_file.name)
            cache.delete(self.filename)

    def clear_cache(self):
        """
        Clear the cache
        :return:
        """
        cache.delete(self.filename)

    def content_parts(self) -> typing.List:
        """
        Get the content parts for the audio
        :return:
        """
        return [self.file]
