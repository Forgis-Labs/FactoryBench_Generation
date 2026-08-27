"""Agentic baseline for FactoryBench.

A GPT-5.1-driven ReAct agent equipped with four general-purpose tools —
manual RAG, signal statistics, forecaster, and a Python sandbox — that
serves as a stronger upper bound than the zero-shot panel.

The knowledge graph is intentionally *not* exposed as a tool: it is the
same catalogue L4 gold answers are drawn from, so exposing it would be
ground-truth leakage.
"""
