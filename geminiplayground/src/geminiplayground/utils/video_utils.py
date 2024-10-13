from io import BytesIO
from pathlib import Path
import typing
import os
import math
import cv2
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType
from tqdm import tqdm


class VideoUtils:
    @staticmethod
    def extract_video_frames(
            video_path: typing.Union[str, Path], output_dir: typing.Union[str, Path]
    ) -> list:
        """
        Extract frames from the video
        :return:
        """
        output_dir = Path(output_dir)
        video_path = Path(video_path)
        vidcap = cv2.VideoCapture(str(video_path))
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        duration = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        video_file_name = video_path.stem

        frame_count = 0  # Initialize a frame counter
        count = 0
        frames_files = []
        with tqdm(
                total=math.ceil(duration), unit="sec", desc="Extracting frames"
        ) as pbar:
            while True:
                ret, frame = vidcap.read()
                if not ret:
                    break
                if count % int(fps) == 0:  # Extract a frame every second
                    frame_count += 1
                    file_name_prefix = os.path.basename(video_file_name).replace(
                        ".", "_"
                    )
                    frame_prefix = "_frame"
                    frame_image_filename = (
                        f"{file_name_prefix}{frame_prefix}{frame_count:04d}.jpg"
                    )
                    frame_image_path = output_dir.joinpath(frame_image_filename)
                    frames_files.append(Path(frame_image_path))
                    cv2.imwrite(str(frame_image_path), frame)
                    pbar.update(1)
                count += 1
        vidcap.release()
        return frames_files

    @staticmethod
    def extract_video_frame_count(video_path: typing.Union[str, Path]) -> int:
        """
        Extract the number of frames in a video
        :param video_path: The path to the video
        :return: The number of frames in the video
        """
        video_path = Path(video_path)
        vidcap = cv2.VideoCapture(str(video_path))
        num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        vidcap.release()
        return int(num_frames)

    @staticmethod
    def extract_video_duration(video_path: typing.Union[str, Path]) -> int:
        """
        Extract the duration of a video
        :param video_path: The path to the video
        :return: The duration of the video in seconds
        """
        video_path = Path(video_path)
        vidcap = cv2.VideoCapture(str(video_path))
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        duration = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        vidcap.release()
        return int(duration)

    @staticmethod
    def extract_video_frame_at_t(
            video_path: typing.Union[str, Path], timestamp_seconds: int
    ) -> PILImageType:
        """
        Extract a frame at a specific timestamp
        :param video_path: The path to the video
        :param timestamp_seconds: The timestamp in seconds
        :return:
        """
        video_path = Path(video_path)
        vidcap = cv2.VideoCapture(str(video_path))
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frame_number = int(fps * timestamp_seconds)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = vidcap.read()
        if not ret:
            raise ValueError(
                f"Could not extract frame at timestamp {timestamp_seconds}"
            )
        vidcap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(frame)

    @classmethod
    def create_video_thumbnail(
            cls,
            video_path: typing.Union[str, Path],
            thumbnail_size: tuple = (128, 128),
            t=0,
    ) -> PILImageType:
        """
        Create a thumbnail for a video
        :param t: The timestamp in seconds
        :param video_path: The path to the video
        :param thumbnail_size: The size of the thumbnail
        :return:
        """
        # Extract the first frame from the video
        first_frame = cls.extract_video_frame_at_t(video_path, t)
        # Create a thumbnail from the first frame
        first_frame.thumbnail(thumbnail_size)
        first_frame = first_frame.convert("RGB")
        thumbnail_bytes = BytesIO()
        first_frame.save(thumbnail_bytes, format="JPEG")
        thumbnail_bytes.seek(0)
        return PILImage.open(thumbnail_bytes)

    @staticmethod
    def seconds_to_time_string(seconds):
        """Converts an integer number of seconds to a string in the format '00:00'.
        Format is the expected format for Gemini 1.5.
        """
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def get_timestamp_seconds(filename, prefix):
        """Extracts the frame count (as an integer) from a filename with the format
        'output_file_prefix_frame0000.jpg'.
        """
        parts = filename.split(prefix)
        if len(parts) != 2:
            return None  # Indicate that the filename might be incorrectly formatted

        frame_count_str = parts[1].split(".")[0]
        return int(frame_count_str)

    @staticmethod
    def get_output_file_prefix(filename, prefix):
        """Extracts the output file prefix from a filename with the format
        'output_file_prefix_frame0000.jpg'.
        """
        parts = filename.split(prefix)
        if len(parts) != 2:
            return None  # Indicate that the filename might be incorrectly formatted

        return parts[0]
