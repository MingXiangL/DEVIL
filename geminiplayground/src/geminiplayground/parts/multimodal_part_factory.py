import mimetypes
from pathlib import Path
from typing import Union

from .git_repo import GitRepo
from .image_part import ImageFile
from .video_part import VideoFile
from .audio_part import AudioFile
from .pdf_part import PdfFile
from ..utils import GitUtils


class MultimodalPartFactory:
    """
    Factory class to create different types of multimodal parts based on file MIME types.
    Supports creation of image, video, audio, PDF files, and Git repository objects.
    """

    @staticmethod
    def from_path(path: Union[str, Path], **kwargs):
        """
        Creates a multimodal part from a given file system path.

        Args:
            path (Union[str, Path]): The file system path to the multimodal part.
            **kwargs: Additional keyword arguments for the part constructors.

        Returns:
            An instance of a part class based on the file type.

        Raises:
            ValueError: If the file type is unsupported or if the path does not exist.
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            mime_type = mimetypes.guess_type(path.as_posix())[0]
            if mime_type is None:
                raise ValueError(f"Unsupported or unknown file type at: {path}")

            if "image" in mime_type:
                return ImageFile(path, **kwargs)
            elif "video" in mime_type:
                return VideoFile(path, **kwargs)
            elif "audio" in mime_type:
                return AudioFile(path, **kwargs)
            elif "pdf" in mime_type:
                return PdfFile(path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {path}")

        elif path.is_dir():
            if GitUtils.folder_contains_git_repo(str(path)):
                return GitRepo.from_folder(path, **kwargs)

        raise ValueError(f"Unsupported directory content at: {path}")
