"""The data shapes a run produces.

Deliberately thin: this re-exports nothing. The private codebase this came from
had a `schemas/__init__.py` that imported all five modules eagerly, which made
every importer of any one shape pay for all of them and was the main reason the
import graph was hard to reason about. Import the module you need.
"""
