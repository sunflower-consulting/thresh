"""A deliberately vulnerable chat application, bundled so thresh ships its target.

WHY THE TOOL CARRIES ITS OWN TARGET.

Every comparable tool needs you to supply something to attack, which means the
first run is gated behind having an endpoint, a key and a config file. That is a
larger obstacle than it sounds: it is the difference between evaluating a tool in
a minute and evaluating it on a Thursday when there is time.

It also makes the tool testable against a known answer. This lab scores itself,
so a run can be checked against what the target believes happened rather than
only against what the tool reports. A measurement tool whose own correctness
rests on reading its own output is not measuring anything.

The brain is a scripted stub by default: no model, no GPU, no network. The
private codebase this came from defaulted the other way, which is right for a
benchmark and wrong for a package whose headline is that it runs with nothing
installed.
"""
from sunflower_thresh.lab.server import serve

__all__ = ["serve"]
