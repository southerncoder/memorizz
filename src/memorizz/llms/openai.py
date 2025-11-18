import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import openai

from .llm_provider import LLMProvider

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    pass


class OpenAI(LLMProvider):
    """
    A class for interacting with the OpenAI API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        context_window_tokens: Optional[int] = None,
    ):
        """
        Initialize the OpenAI client.

        Parameters:
        -----------
        api_key : str
            The API key for the OpenAI API.
        model : str, optional
            The model to use for the OpenAI API.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.context_window_tokens = (
            context_window_tokens or self._infer_context_window_tokens(model)
        )
        self._last_usage: Optional[Dict[str, int]] = None

    def _infer_context_window_tokens(self, model: str) -> int:
        """Best-effort mapping of well-known OpenAI models to their context window."""
        normalized = model.lower() if model else ""
        context_map = {
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "gpt-4.1": 128_000,
            "gpt-4.1-mini": 128_000,
            "gpt-4.1-turbo": 128_000,
            "gpt-4-turbo": 128_000,
            "gpt-4.1-preview": 128_000,
            "gpt-4o-mini-high": 128_000,
            "gpt-4o-mini-low": 128_000,
            "gpt-4o-mini-4096": 4_096,
            "gpt-4o-realtime-preview": 128_000,
        }
        return context_map.get(normalized, 128_000)

    def get_config(self) -> Dict[str, Any]:
        """Returns a serializable configuration for the OpenAI provider."""
        return {"provider": "openai", "model": self.model}

    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """
        Get the metadata for a tool.

        Parameters:
        -----------
        func : Callable
            The function to get the metadata for.

        Returns:
        --------
        Dict[str, Any]
        """
        # We'll import ToolSchemaType here to avoid circular imports
        from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType

        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        func_name = func.__name__

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert metadata augmentation assistant specializing in JSON schema discovery "
                "and documentation enhancement.\n\n"
                f"**IMPORTANT**: Use the function name exactly as provided (`{func_name}`) and do NOT rename it."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Generate enriched metadata for the function `{func_name}`.\n\n"
                f"- Docstring: {docstring}\n"
                f"- Signature: {signature}\n\n"
                "Enhance the metadata by:\n"
                "• Expanding the docstring into a detailed description.\n"
                "• Writing clear natural‐language descriptions for each parameter, including type, purpose, and constraints.\n"
                "• Identifying which parameters are required.\n"
                "• (Optional) Suggesting example queries or use cases.\n\n"
                "Produce a JSON object that strictly adheres to the ToolSchemaType structure."
            ),
        }

        response = self.client.responses.parse(
            model=self.model, input=[system_msg, user_msg], text_format=ToolSchemaType
        )

        return response.output_parsed

    def augment_docstring(self, docstring: str) -> str:
        """
        Augment the docstring with an LLM generated description.

        Parameters:
        -----------
        docstring : str
            The docstring to augment.

        Returns:
        --------
        str
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Augment the docstring {docstring} by adding more details and examples.",
        )

        return response.output_text

    def generate_queries(self, docstring: str) -> List[str]:
        """
        Generate queries for the tool.

        Parameters:
        -----------
        docstring : str
            The docstring to generate queries for.

        Returns:
        --------
        List[str]
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Generate queries for the docstring {docstring} by adding some examples of queries that can be used to leverage the tool.",
        )

        return response.output_text

    def generate_text(self, prompt: str, instructions: str = None) -> str:
        """
        Generate text using OpenAI's API.

        Parameters:
            prompt (str): The prompt to generate text from.
            instructions (str): The instructions to use for the generation.

        Returns:
            str: The generated text.
        """
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=prompt
        )

        return response.output_text

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """
        Generate a response using OpenAI's chat completions API.

        Parameters:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
                Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            tools (Optional[List[Dict[str, Any]]]): List of tool definitions for function calling.
            tool_choice (str): Controls which (if any) function is called ("auto", "none", or specific function).

        Returns:
            Any: Either a string (final response) or the full response object (if tool calls are present).
        """
        kwargs = {"model": self.model, "messages": messages}

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        self._last_usage = self._extract_usage(response)

        # If there are tool calls, return the full response object
        if response.choices[0].message.tool_calls:
            return response

        # Otherwise return just the text content
        return response.choices[0].message.content

    def _extract_usage(self, response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        try:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception:
            return None

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens
