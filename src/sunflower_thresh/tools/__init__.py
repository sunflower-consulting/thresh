"""The chat transport, and the base class it shares with nothing else.

The package it came from had a wide tool surface: web request builders, a
headless browser, OT protocol clients, shell runners. None of that is here, so
`Tool` has exactly one subclass. The abstraction is kept anyway because it is
what carries the action class and the evidence kind to the gate, and inlining it
would move that wiring into the transport where it is easier to get wrong.
"""
