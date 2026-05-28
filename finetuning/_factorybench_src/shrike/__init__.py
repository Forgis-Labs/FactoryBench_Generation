"""Shrike — Time Series Understanding via Discrete Tokenization.

A backbone-agnostic framework that turns any decoder-only LLM into a
time-series reasoner through discrete VQ-VAE tokenization.

Quick start::

    import shrike as hy

    # Load a pretrained model
    model = hy.Shrike.from_pretrained("checkpoints/best_model.pt",
                                        totem_ckpt="checkpoints/totem.pt",
                                        llm_id="Qwen/Qwen3-4B")

    # Analyze a signal
    result = model.analyze(signal, question="What is the trend?")
    print(result)

    # Forecast (Chameleon-style: generates codes, decodes to values)
    forecast = model.forecast(signal, horizon=64)
    print(forecast.values)

Submodules:
    shrike.model   — Shrike model + TOTEM tokenizer
    shrike.data    — Dataset loading + building
    shrike.eval    — Evaluation + sensitivity test
    shrike.train   — Training loops
"""

__version__ = "0.1.0"

# Convenience re-exports so users can write:
#   from shrike import Shrike, ShrikeConfig
#   or: import shrike as hy; hy.Shrike(...)
#
# Lazy imports: avoids pulling in peft/transformers when only
# the tokenizer subpackage is needed (e.g., tokenizer training on SageMaker).
def __getattr__(name):
    if name == "Shrike" or name == "ShrikeConfig":
        from shrike.model.shrike import Shrike, ShrikeConfig
        return Shrike if name == "Shrike" else ShrikeConfig
    if name == "TOTEMTokenizer":
        from shrike.model.totem import TOTEMTokenizer
        return TOTEMTokenizer
    raise AttributeError(f"module 'shrike' has no attribute {name!r}")
