from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> dict:
        """Return a dict with keys: text (str), usage (optional dict)."""
        ...
