import inspect
import typing

from google.generativeai.types import File, FunctionDeclaration
from datetime import datetime, timezone
from pathlib import Path
import os
import re

from pydantic import Field, BaseModel, create_model


class LibUtils:
    """
    A utility class for the playground
    """

    @staticmethod
    def get_lib_home() -> Path:
        """
        Get the cache directory for the playground
        :return:
        """
        cache_dir = os.environ.get(
            "GEMINI_PLAYGROUND_HOME", Path.home() / ".gemini_playground"
        )
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @staticmethod
    def get_uploaded_file_exp_date_delta_t(uploaded_file: File) -> float:
        """
        Get the delta time
        :param uploaded_file: the uploaded file
        """
        now = datetime.now(timezone.utc)
        future = uploaded_file.expiration_time
        delta_t = future - now
        delta_t = delta_t.total_seconds()
        return delta_t

    @staticmethod
    def split_and_label_prompt_parts_from_string(input_string):
        """
        Split and label the prompt parts from a string
        """
        # This regex looks for substrings that are either inside brackets (considered files) or are not brackets and
        # commas (considered text).
        pattern = r"\[([^\]]+)\]|([^[\]]+)"
        # Find all matches of the pattern in the input string
        matches = re.findall(pattern, input_string)
        # Initialize an empty list to store the result
        result = []

        for match in matches:
            file, text = match  # Unpack the tuple

            # Check if the match is considered a file (inside brackets) or text (outside brackets)
            if file:
                result.append({"type": "multimodal", "value": file.strip()})
            elif text.strip():  # Ensure text is not just whitespace
                result.append({"type": "text", "value": text.strip()})

        return result

    @staticmethod
    def has_complete_type_hints(func):
        """
        Check if a function has complete type hints for all parameters and the return type
        """
        # Get the signature of the function
        signature = inspect.signature(func)

        # Check type annotations for all parameters
        params = signature.parameters

        for name, param in params.items():
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter {name} is missing a type hint.")

        # Check type annotation for the return type
        if signature.return_annotation is inspect.Signature.empty:
            raise TypeError("Return type is missing a type hint.")

        return True

    @staticmethod
    def func_to_pydantic(func) -> BaseModel:
        """
        Convert a function to a Pydantic model
        """
        model_fields = {}
        func_params = inspect.signature(func).parameters
        for k, v in func_params.items():
            if k == "self":
                continue
            input_annot = v.annotation

            field_default = None if v.default == inspect.Parameter.empty else v.default
            field_description = ""
            if input_annot is inspect.Parameter.empty and field_default is None:
                field_type = typing.Any
            elif input_annot is inspect.Parameter.empty:
                field_type = type(field_default)
            else:
                field_type = input_annot
                annot_origin = typing.get_origin(input_annot)
                annot_args = typing.get_args(input_annot)
                if annot_origin is typing.Annotated:
                    field_description = annot_args[1]
            model_fields[k] = (
                field_type,
                Field(field_default, description=field_description),
            )

        model_desc = func.__doc__
        model_title = func.__name__
        model_args = {**model_fields, **{"__doc__": model_desc}}
        return create_model(model_title, **model_args)

    @classmethod
    def func_to_tool(cls, func):
        """
        Convert a function to a Gemini function definition
        """
        func_properties = cls.func_to_pydantic(func).schema()["properties"]
        func_properties = {
            k: {"type": v["type"], "description": v["description"]}
            for k, v in func_properties.items()
        }
        return FunctionDeclaration(
            name=func.__name__,
            description=func.__doc__,
            parameters={
                "type": "object",
                "properties": func_properties,
            },
        )

    @staticmethod
    def normalize_prompt(prompt):
        """
        Normalize the prompt.
        :param prompt:
        :return:
        """
        from geminiplayground.parts import MultimodalPart

        normalized_prompt = []

        if isinstance(prompt, str):
            prompt = [prompt]

        for part in prompt:
            if isinstance(part, str):
                normalized_prompt.append(part)
            elif isinstance(part, MultimodalPart):
                content_parts = part.content_parts()
                normalized_prompt.extend(content_parts)
            elif isinstance(part, File):
                normalized_prompt.append(part)
            else:
                raise ValueError(f"Invalid prompt part: {part}")
        return normalized_prompt
