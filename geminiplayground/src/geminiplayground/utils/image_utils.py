import typing
from pathlib import Path

from PIL import Image as PILImage
from PIL.Image import Image as PILImageType


class ImageUtils:
    """
    Utility class for working with images
    """

    @staticmethod
    def create_image_thumbnail(
            image_path: typing.Union[str, Path], thumbnail_size: tuple = (128, 128)
    ) -> PILImageType:
        """
        Create a thumbnail for an image
        :param pil_image: The image to create a thumbnail for
        :param thumbnail_size: The size of the thumbnail
        :return:
        """
        pil_image = PILImage.open(image_path)
        pil_image.thumbnail(thumbnail_size)
        if pil_image.mode == "RGBA":
            background = PILImage.new("RGB", pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        return pil_image
