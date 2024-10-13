from abc import ABC, abstractmethod


class MultimodalPart(ABC):
    """
    Abstract class for multimodal part
    """

    def upload(self, **kwargs):
        """
        Upload the multimodal part
        :param kwargs:
        :return:
        """
        ...

    def clear_cache(self, **kwargs):
        """
        Clear the multimodal part
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def content_parts(self, **kwargs):
        """
        transform a multimodal part into a list of content parts
        :param kwargs:
        :return:
        """
        raise NotImplementedError
