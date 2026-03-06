import os
from dotenv import load_dotenv
from .base import ModelAdapter

load_dotenv()


class AzureOpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        deployment: str,
        api_version: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
    ):
        from openai import AzureOpenAI
        
        self.client = AzureOpenAI(
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
        )
        self.deployment = deployment

    def generate(self, prompt: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (resp.choices[0].message.content or "")
            usage = getattr(resp, "usage", None)
            usage_dict = {}
            if usage:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return {"text": text, "usage": usage_dict}
        except Exception as e:
            return {"text": f"ERROR: azure generation failed: {type(e).__name__}: {e}"[:500], "usage": {}}
