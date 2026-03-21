from __future__ import annotations

import os

from .gpt import OpenAIWrapper


class DeepOCRAPI(OpenAIWrapper):
    """OpenAI-compatible API wrapper for DeepOCR pipeline endpoint.

    Credentials and endpoint are provided only via environment variables:
      - DEEPOCR_API_BASE
      - DEEPOCR_API_KEY
    """

    is_api: bool = True

    def __init__(
        self,
        model: str = "deepocr",
        retry: int = 5,
        verbose: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0,
        timeout: int = 300,
        max_tokens: int = 2048,
        img_size: int = -1,
        img_detail: str = "high",
        **kwargs,
    ):
        api_base = os.getenv("DEEPOCR_API_BASE", "")
        api_key = os.getenv("DEEPOCR_API_KEY", "")
        if not api_base or not api_key:
            raise ValueError(
                "DEEPOCR_API_BASE and DEEPOCR_API_KEY must be set in the environment."
            )

        super().__init__(
            model=model,
            retry=retry,
            key=api_key,
            verbose=verbose,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout=timeout,
            api_base=api_base,
            max_tokens=max_tokens,
            img_size=img_size,
            img_detail=img_detail,
            **kwargs,
        )
