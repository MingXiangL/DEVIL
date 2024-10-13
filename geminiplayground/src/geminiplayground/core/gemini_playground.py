import typing
from pathlib import Path

from google.generativeai.types import Tool

from .gemini_client import GeminiClient
from geminiplayground.utils import LibUtils
from pydantic import BaseModel
import logging

logger = logging.getLogger("rich")


class ToolCall(BaseModel):
    """
    A message from the Gemini model.
    """

    tool_name: typing.Any
    tool_result: typing.Any


class Message(BaseModel):
    """
    A message from the Gemini model.
    """

    text: str


class ChatSession:
    """
    A chat session with the Gemini model.
    """

    @property
    def history(self):
        """
        Get the chat history.
        """
        return self.chat.history

    def __init__(self, model: str, history: list, toolbox: dict, **kwargs):
        self.model = model
        self.toolbox = toolbox
        tools_def = [
            Tool(
                function_declarations=[
                    LibUtils.func_to_tool(tool_func)
                    for tool_name, tool_func in self.toolbox.items()
                ]
            )
        ]
        gemini_client = kwargs.pop("gemini_client", GeminiClient())
        self.chat = gemini_client.start_chat(
            model=model,
            history=history,
            tools=tools_def, **kwargs)

    def clear_history(self):
        """
        Clear the chat history.
        """
        self.chat.history.clear()

    def send_message(self, message: str, stream: bool = True, **kwargs) -> typing.Generator:
        """
        Send a message to the chat session.
        """
        normalized_message = LibUtils.normalize_prompt(message)
        response = self.chat.send_message(normalized_message, stream=stream, **kwargs)
        for response_chunk in response:
            for part in response_chunk.parts:
                if fn := part.function_call:
                    fun_name = fn.name
                    fun_args = dict(fn.args)
                    result = self.toolbox[fun_name](**fun_args)
                    yield ToolCall(tool_name=fun_name, tool_result=result)
                else:
                    yield Message(text=part.text)
        response.resolve()


class GeminiPlayground:
    """
    A playground for testing the Gemini model.
    """

    def __init__(
            self, model: str
    ):
        self.model = model
        self.toolbox = {}

    def add_file(self, file: typing.Union[str, Path]):
        """
        Add a file to the playground.
        @param file: The file to add
        """
        raise NotImplementedError("Not implemented yet")

    def add_tool(self, tool):
        """
        Add a tool to the playground.
        """
        self.tool(tool)

    def tool(self, func):
        """
        A decorator to add a tool to the playground.
        """
        # check for docstring
        if not func.__doc__:
            raise ValueError(f"Function {func.__name__} must have a docstring")
        # check for functions hints
        if not LibUtils.has_complete_type_hints(func):
            raise ValueError(f"Function {func.__name__} must have complete type hints")
        self.toolbox[func.__name__] = func

    def start_chat(self, history: list = None, **kwargs):
        """
        Start a chat session with the playground.
        """
        return ChatSession(self.model, history, self.toolbox, **kwargs)
