from typing import Callable


def simple_preprocessor(text: str) -> str:
    """Lightweight text preprocessor used before vectorization.

    - Lowercases
    - Strips leading/trailing whitespace
    - Leaves tokenization to the vectorizer
    """
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


__all__ = ["simple_preprocessor"]
