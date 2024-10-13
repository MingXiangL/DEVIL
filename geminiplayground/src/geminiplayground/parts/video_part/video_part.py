import json
import logging
import time
import typing

from pydantic import BaseModel

from geminiplayground.catching import cache
from geminiplayground.core import GeminiClient
from .. import MultimodalPart
from geminiplayground.utils import FileUtils, LibUtils
from pathlib import Path
import google.generativeai as genai

logger = logging.getLogger("rich")


class VideoFile(MultimodalPart):
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

    def check_status(self):
        """
        Check the status of the file
        :return:
        """
        remote_file_name = self.file.name
        file = self.gemini_client.get_file(remote_file_name)
        return file.state.name

    def wait(self):
        """
        Wait until the file is ready
        :return:
        """
        logger.info(f"Waiting for the file {self.filename} to be ready")
        remote_file_name = self.file.name
        file = self.gemini_client.get_file(remote_file_name)
        while file.state.name == "PROCESSING":
            time.sleep(10)
            file = self.gemini_client.get_file(remote_file_name)
        if file.state.name == "FAILED":
            raise Exception("File upload failed")

    @property
    def file(self):
        """
        Get the files
        :return:
        """
        if cache.get(self.filename):
            logger.info(f"Getting image file {self.filename} from cache")
            cached_file = cache.get(self.filename)
            return cached_file

        logger.info(f"Uploading image file {self.filename}")
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
        if self.check_status() == "PROCESSING":
            self.wait()
        return [self.file]

    def extract_keyframes(self, model: str = "models/gemini-1.5-flash-latest"):
        """
        Get the timeline of the video
        :return:
        """

        class VideoKeyFrame(BaseModel):
            """
            Video key frame
            """

            timespan: str
            description: str

        system_instruction = """you are a video processing system, follow the instructions and extract the key frames 
        in the provided video, your response should include a description of max 100 characters and the timespan of 
        each key frame"""
        prompt = ["return the key frames in the following video", self.file]
        raw_response = self.gemini_client.generate_response(
            model,
            prompt,
            stream=False,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[VideoKeyFrame],
            ),
        )
        response = json.loads(raw_response.text)
        return response["key_frames"]
