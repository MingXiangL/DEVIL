import typing
from pathlib import Path
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType
import fitz


class PDFUtils:
    """
    Utility class for working with PDF files
    """

    @staticmethod
    def create_pdf_thumbnail(
        pdf_path: typing.Union[str, Path],
        thumbnail_size: tuple = (128, 128),
        zoom: float = 0.2,
    ) -> PILImageType:
        """
        Create a thumbnail for a PDF
        :param pdf_path: The path to the PDF
        :param thumbnail_size: The size of the thumbnail
        :param zoom: The zoom factor for the image
        :return:
        """
        # Convert the first page of the PDF to an image
        # Open the provided PDF file
        document = fitz.open(pdf_path)

        # Select the first page
        page = document[0]

        # Set the zoom factor for the image
        mat = fitz.Matrix(zoom, zoom)

        # Render page to an image (pixmap)
        pix = page.get_pixmap(matrix=mat)

        # Save the pixmap as an image file
        image = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Create a thumbnail from the image
        image.thumbnail(thumbnail_size)
        return image
