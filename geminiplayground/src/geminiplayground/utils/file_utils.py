import shutil
import ssl
from urllib.parse import urlparse
import urllib.request
import validators
from pathlib import Path
import os
import tempfile
from contextlib import contextmanager
import typing
from urllib.error import HTTPError

ssl._create_default_https_context = ssl._create_unverified_context


class FileUtils:
    """
    File utilities
    """

    @classmethod
    def clear_folder(cls, pth: typing.Union[str, Path]):
        """
        Recursively remove a directory and its contents
        :param pth:
        :return:
        """
        pth = Path(pth)
        if not pth.exists():
            return
        for child in pth.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child, ignore_errors=True)

    @classmethod
    @contextmanager
    def onflyTemporaryDirectory(cls, **kwargs):
        """
        Create a temporary directory
        @param suffix: the suffix of the directory
        """
        name = tempfile.mkdtemp(**kwargs)
        try:
            yield name
        finally:
            cls.rm_tree(name)

    @staticmethod
    @contextmanager
    def onflyTemporaryFile(**kwargs):
        """
        Create a temporary file
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, **kwargs)
        try:
            yield tmp
        finally:
            tmp.close()
            os.unlink(tmp.name)

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        converts gcs uri to url for image display.
        @param url: the url of the file
        """
        url_parts = urlparse(url)
        scheme = url_parts.scheme
        if scheme == "gs":
            return "https://storage.googleapis.com/" + url.replace("gs://", "").replace(
                " ", "%20"
            )
        elif scheme in ["http", "https"]:
            return url
        raise Exception("Invalid scheme")

    @classmethod
    @contextmanager
    def get_path_from_url(cls, url: str) -> typing.ContextManager:
        """
        copy the file from the url to a temporary file and return the path as a context manager
        @param url: the url of the file
        """
        http_uri = cls.normalize_url(url)
        try:
            assert validators.url(http_uri), "invalid url"
            resp = urllib.request.urlopen(url, timeout=30)
            file_name = cls.get_file_name_from_path(url)
            file_extension = Path(file_name).suffix
            file_name_stem = Path(file_name).stem
            with cls.onflyTemporaryFile(
                    prefix=file_name_stem, suffix=file_extension
            ) as temp_file:
                with open(temp_file.name, "w+b") as f:
                    f.write(resp.read())
                yield temp_file.name
        except HTTPError as err:
            if err.strerror == 404:
                raise Exception("Image not found")
            elif err.code in [403, 406]:
                raise Exception("Forbidden image, it can not be reached")
            else:
                raise

    @staticmethod
    @contextmanager
    def get_path_from_local(path: typing.Union[str, Path]) -> typing.ContextManager:
        """
        return the path of the file as a context manager
        """
        yield path

    @classmethod
    def solve_file_path(
            cls, uri_or_path: typing.Union[str, Path]
    ) -> typing.ContextManager:
        """
        return the path of the file
        """
        uri_or_path = str(uri_or_path)
        return (
            cls.get_path_from_url(uri_or_path)
            if validators.url(uri_or_path)
            else cls.get_path_from_local(uri_or_path)
        )

    @staticmethod
    def get_file_name_from_path(path: typing.Union[str, Path], include_extension=True):
        """
        Get the file name from a path
        :param path: The path to the file
        :param include_extension: Include the extension in the file name
        :return:
        """
        path = str(path)
        if validators.url(path):
            file_path = Path(urlparse(path).path)
        else:
            file_path = Path(path)
        if include_extension:
            return file_path.name
        return file_path.stem

    @staticmethod
    def humanize_file_size(size_in_bytes: float) -> str:
        """
        Convert size in bytes to human readable format
        :param size_in_bytes: The size in bytes
        :return: Human readable size
        """
        # Define the threshold for each size unit
        units = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        size = size_in_bytes
        unit_index = 0

        while size > 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        # Format the size with two decimal places and the appropriate unit
        return f"{size:.2f} {units[unit_index]}"

    @staticmethod
    def get_file_size(file_path: typing.Union[str, Path]):
        """
        Get the size of a file in bytes
        """
        # Use os.path.getsize() to get the file size in bytes
        size_in_bytes = os.path.getsize(file_path)
        return size_in_bytes
