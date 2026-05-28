"""Shrike model + TOTEM tokenizer."""

# Lazy imports — avoids pulling in peft/transformers when only
# the tokenizer subpackage is needed (e.g., tokenizer training on SageMaker).
def __getattr__(name):
    if name in ("Shrike", "ShrikeConfig"):
        from .shrike import Shrike, ShrikeConfig
        return Shrike if name == "Shrike" else ShrikeConfig
    if name == "TOTEMTokenizer":
        from .totem import TOTEMTokenizer
        return TOTEMTokenizer
    raise AttributeError(f"module 'shrike.model' has no attribute {name!r}")

__all__ = ["Shrike", "ShrikeConfig", "TOTEMTokenizer"]
