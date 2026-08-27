"""Tool implementations for the agentic baseline.

Each tool exposes a plain Python callable + an OpenAI function-calling
schema (returned by ``spec()``). The agent loop registers all tools it
wants to expose and dispatches calls by name.
"""

from .signal_stats import SignalStatsTool
from .dynamics import DynamicsTool
from .forecast import ForecastTool
from .python_sandbox import PythonSandboxTool
from .manual_rag import ManualRAGTool

__all__ = [
    "SignalStatsTool",
    "DynamicsTool",
    "ForecastTool",
    "PythonSandboxTool",
    "ManualRAGTool",
]
