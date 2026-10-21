"""The policy gate: the request and decision shapes, the engine, the kill switch.

Deliberately thin: this re-exports nothing, for the reason `schemas/__init__.py`
gives at greater length. Import the module you need.

It used to re-export seven names, and nothing in the repository imported any of
them -- all twenty call sites name a submodule directly. The facade was not free,
though. `tools/base.py` and `tools/conversation.py` need `policy.decision`, a
125-line module of request shapes, and importing it ran this file first, which
pulled in the engine, the kill switch, the audit log and the target schema with
pydantic behind them. A package that advertises `thresh --help` as cheap cannot
afford a facade nobody calls.
"""
