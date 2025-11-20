from .base import ModelAdapter


class MockAdapter(ModelAdapter):
    def generate(self, prompt: str) -> dict:
        return {"text": "mean=0 min=0 max=0", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
