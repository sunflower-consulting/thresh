"""One structured log for everything a run does -- human-readable and parseable.

A run writes a single append-only JSONL stream of ``LogEvent`` records under
``runs/<run-id>/events.jsonl``, plus a ``manifest.json`` summary. Every component
(orchestrator, campaign, validation, the gate decisions a run observes) emits the
*same* event shape, so one grep or one ``harness log <run-id>`` reconstructs
exactly what happened -- which is what you need when something breaks on your
side or a customer's. JSONL is trivially machine-parseable; the renderer turns
the same records into aligned, colourised lines for a human.

Schema (one JSON object per line):
    {ts, run_id, seq, level, actor, event, message, data}
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from sunflower_thresh import termstyle as ts
from sunflower_thresh.redact import redact

LEVELS = ("debug", "info", "warn", "error")


class LogEvent(BaseModel):
    ts: str
    run_id: str
    seq: int
    level: str = "info"
    actor: str = ""           # orchestrator | campaign | gate | validation | tool | model
    event: str = ""           # run_start | decision | policy | tool_result | finding | budget | run_end | error
    message: str = ""
    data: dict = Field(default_factory=dict)


#: Characters JSON does NOT escape but Python's `str.splitlines()` -- and many
#: other line splitters -- treat as line breaks. A tool result carrying one (any
#: JavaScript bundle may, and crawling a web app reads plenty) was written as a
#: single valid JSONL line and read back as two unparseable fragments, so the
#: event vanished from the view, from `harness log`, and from every reader of
#: the run. Because a torn line is skipped rather than raised, the loss was
#: silent and took the whole run's log with it.
_LINE_BREAKERS = {"\u2028": "\\u2028", "\u2029": "\\u2029", "\u0085": "\\u0085"}


def _one_line(blob: str) -> str:
    """One JSON record, guaranteed to occupy exactly one line."""
    for raw, escaped in _LINE_BREAKERS.items():
        if raw in blob:
            blob = blob.replace(raw, escaped)
    return blob


def _redact_tree(value, redactor):
    """Redact every string inside a nested structure, leaving shape and keys alone.

    Walks rather than redacting the serialised line, because a pattern applied to
    whole JSON can match across two field boundaries and produce a line that no
    longer parses. Keys are left untouched: they are field names chosen by this
    codebase, not captured data.

    The redactor passed here should be the SAME object the evidence store holds.
    `make_redactor` keeps its protected values in a closure, so one registration
    of a canary minted mid-run protects it in both places; two redactors would
    mean a token that survives into the evidence and is destroyed in the log.
    """
    keep = tuple(getattr(redactor, "kept", ()) or ())
    if isinstance(value, str):
        return redact(value, keep=keep)[0]
    if isinstance(value, dict):
        return {k: _redact_tree(v, redactor) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_tree(v, redactor) for v in value)
    return value


class RunLog:
    """Append-only event writer for one run. Thread-light (single run owner)."""

    def __init__(self, run_id: str, root: str | Path = "runs", redactor=None):
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._seq = 0
        #: Applied to `message` and `data` before the line is written. The run
        #: log is not a scratch file: it is the timestamped record of what was
        #: sent, which is the artefact a client's own monitoring gets compared
        #: against, so it leaves this machine. A support bot's replies are made
        #: of customer data and a captured Authorization header is a live
        #: credential, and neither belongs in a file that gets handed over.
        self.redactor = redactor
        # Taking a sequence number and appending the line are ONE operation.
        # `self._seq += 1` is a read, an add and a write, and two workers that
        # interleave between them get the same number -- `seq` is what orders
        # the whole event stream, so the view and `harness log` would both be
        # reading a history that never happened. At the default 5ms switch
        # interval this is invisible; with the interval turned down it is 1,632
        # duplicates in 7,200 events, which is what a loaded machine or a
        # free-threaded interpreter will eventually do on its own.
        self._lock = threading.Lock()
        #: Called with each event after it is written. Nothing in the harness
        #: needs it; the benchmark hangs its periodic scoreboard read off it,
        #: because the loop is already emitting and a sampler that rate-limits
        #: itself then costs one cheap GET every twenty seconds without a
        #: thread of its own. Never allowed to break a run.
        self.on_event = None

    def emit(self, event: str, *, actor: str = "", level: str = "info",
             message: str = "", **data) -> LogEvent:
        if self.redactor is not None:
            message = _redact_tree(message, self.redactor)
            data = _redact_tree(data, self.redactor)
        with self._lock:
            ev = LogEvent(ts=datetime.now(UTC).isoformat(), run_id=self.run_id,
                          seq=self._seq, level=level, actor=actor, event=event,
                          message=message, data=data)
            self._seq += 1
            with self.path.open("a") as fh:
                fh.write(_one_line(ev.model_dump_json()) + "\n")
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001,S110 - an observer never breaks a run
                # Deliberately silent and deliberately broad. An observer is a
                # convenience hung off the event stream by a caller; a run that
                # died because somebody's scoreboard sampler raised would lose
                # measurements to a cosmetic failure.
                pass
        return ev

    # --- typed conveniences (all share the one schema) ---
    def run_start(self, **meta) -> LogEvent:
        return self.emit("run_start", actor="orchestrator", message=meta.get("objective", ""), **meta)

    def decision(self, actor, thought, action, tool="", args=None) -> LogEvent:
        return self.emit("decision", actor=actor, message=thought,
                         action=action, tool=tool, args=args or {})

    def policy(self, tool, allowed, deny_code="", target="", action_class="") -> LogEvent:
        """One gate decision.

        `action_class` is the authoritative passive/active/destructive label from the
        request. Without it the bench lamp falls back to guessing from a hardcoded list
        of tool names (`bench/states.py:PASSIVE_TOOLS`), and its own comment says a
        caller that supplies this wins over the guess. The guess cannot be right in
        general -- it is a web-tool list, so every chat verb read as active -- and the
        caller already has the real value.
        """
        return self.emit("policy", actor="gate",
                         level="info" if allowed else "warn",
                         message=f"{'allow' if allowed else 'deny'} {tool}",
                         tool=tool, allowed=allowed, deny_code=deny_code, target=target,
                         action_class=action_class)

    def tool_result(self, tool, summary, *, returncode=None, evidence_refs=None) -> LogEvent:
        return self.emit("tool_result", actor="tool", message=summary, tool=tool,
                         returncode=returncode, evidence_refs=evidence_refs or [])

    def finding(self, finding_id, state, cves=None) -> LogEvent:
        return self.emit("finding", actor="validation",
                         message=f"{finding_id} -> {state}",
                         finding_id=finding_id, state=state, cves=cves or [])

    # --- agent activity ------------------------------------------------------
    # Four events the live view needs and the stream did not carry. They follow
    # the same schema as everything else, so `harness log` renders them and a
    # grep still works; nothing already parsing this stream has to change,
    # because unknown event names were already ignored rather than fatal.
    #
    # A run that nobody is watching pays one appended line per event and no more:
    # the view is a reader, and emitting is unconditional so a run behaves
    # identically whether or not a TUI is attached.

    def agent_spawned(self, agent_id, *, role, model, hub: bool = False,
                      kind: str = "agent", owner: str = "") -> LogEvent:
        """An actor joined the run. `hub` marks the orchestrator itself.

        The view assigns a ring position on first sight and keeps it for the rest
        of the run, so an agent that finishes stays where the operator last saw
        it instead of the ring reshuffling under them.

        `kind="tool"` says this is something an agent INVOKED rather than an
        actor with a life of its own, and `owner` names the agent that invoked
        it. Only agents get a ring position; tools are listed under their owner
        in the table. Without the distinction a single-agent run drew seven
        boxes -- curl, feroxbuster, http_request, login -- and looked exactly
        like a fleet.
        """
        return self.emit("agent_spawned", actor=agent_id,
                         # "feroxbuster on feroxbuster" said nothing twice; a tool whose
            # backend IS its own name just says its name.
            message=(f"{role} on {model}" if model and model != role else str(role)),
                         agent_id=agent_id, role=role, model=model, hub=hub,
                         kind=kind, owner=owner)

    def agent_state(self, agent_id, state, *, task: str = "", detail: str = "") -> LogEvent:
        """`setting up` | `running` | `waiting` | `done` | `error`, plus the task
        the timer is counting. A state change restarts that timer."""
        return self.emit("agent_state", actor=agent_id,
                         level="error" if state == "error" else "info",
                         message=f"{state}{': ' + task if task else ''}",
                         agent_id=agent_id, state=state, task=task, detail=detail)

    def agent_message(self, frm, to, *, kind: str = "", size: int = 0) -> LogEvent:
        """One message between two actors, which is what animates a link.

        Agent-to-agent traffic is routed through the hub, so A->B is emitted as
        it really happens: A->hub then hub->B. The view plays the legs in
        sequence rather than drawing a link that does not exist.
        """
        return self.emit("agent_message", actor=frm,
                         message=f"{frm} -> {to}" + (f" ({kind})" if kind else ""),
                         **{"from": frm}, to=to, kind=kind, size=size)

    def agent_usage(self, agent_id, *, tokens: int = 0, cost: float = 0.0) -> LogEvent:
        """Running totals per agent, for the detail table."""
        return self.emit("agent_usage", actor=agent_id, level="debug",
                         message=f"{tokens} tok", agent_id=agent_id,
                         tokens=tokens, cost=cost)

    def budget(self, reason) -> LogEvent:
        return self.emit("budget", actor="orchestrator", level="warn",
                         message=f"stop: {reason}", reason=reason)

    def error(self, message, **data) -> LogEvent:
        return self.emit("error", actor=data.pop("actor", "orchestrator"),
                         level="error", message=message, **data)

    def run_end(self, **summary) -> LogEvent:
        self.write_manifest(summary)
        return self.emit("run_end", actor="orchestrator",
                         message=summary.get("stop_reason", "done"), **summary)

    def write_manifest(self, summary: dict) -> None:
        manifest = {"run_id": self.run_id, "events": str(self.path), **summary}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


# ------------------------------------------------- run economics ledger
def append_run_record(runs_dir: str | Path, record: dict) -> None:
    """Append one run's economics to the runs ledger (runs/ledger.jsonl)."""
    d = Path(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def summarize_runs(runs_dir: str | Path) -> dict:
    """Aggregate the runs ledger into the business metrics (§15)."""
    p = Path(runs_dir) / "ledger.jsonl"
    rows = [json.loads(line) for line in p.read_text().split("\n") if line.strip()] \
        if p.exists() else []
    runs = len(rows)
    validated = sum(int(r.get("validated", 0)) for r in rows)
    findings = sum(int(r.get("findings", 0)) for r in rows)
    cost = sum(float(r.get("cost_usd", 0.0)) for r in rows)
    tokens = sum(int(r.get("total_tokens", 0)) for r in rows)
    seconds = sum(float(r.get("seconds", 0.0)) for r in rows)
    return {
        "runs": runs, "findings": findings, "validated": validated,
        "total_cost_usd": round(cost, 6), "total_tokens": tokens,
        "total_seconds": round(seconds, 1),
        "cost_per_validated_finding": round(cost / validated, 6) if validated else None,
        "mean_seconds_per_run": round(seconds / runs, 1) if runs else None,
    }


def tally_runs(runs_dir: str | Path) -> dict:
    """Cross-run tally -- the 'results after weeks of work' scoreboard. Aggregates
    the run ledger and the benchmark log into cumulative totals, per-project /
    per-model / per-day breakdowns, and the UNION of every challenge ever solved."""
    import re as _re
    d = Path(runs_dir)

    def _load(name: str) -> list[dict]:
        p = d / name
        return [json.loads(x) for x in p.read_text().split("\n") if x.strip()] if p.exists() else []

    ledger = _load("ledger.jsonl")
    bench = _load("benchmark.jsonl")

    def _day(row: dict) -> str:
        # Run rows carry "at"; benchmark rows carried only "ts", which showed as
        # "unknown" in every per-day breakdown. Accept either.
        at = row.get("at") or row.get("ts") or ""
        if not at:                                  # fall back to run_id's -YYYYMMDD- stamp
            m = _re.search(r"(\d{8})-\d{6}", row.get("run_id", ""))
            if m:
                at = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
        return at[:10] or "unknown"

    by_project: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    by_day: dict[str, dict] = {}
    for r in ledger:
        for bucket, key in ((by_project, r.get("project", "?")), (by_model, r.get("model", "?")),
                            (by_day, _day(r))):
            b = bucket.setdefault(key, {"runs": 0, "findings": 0, "validated": 0,
                                        "missed": 0, "cost_usd": 0.0})
            b["runs"] += 1
            b["findings"] += int(r.get("findings", 0))
            b["validated"] += int(r.get("validated", 0))
            b["missed"] += int(r.get("missed", 0))
            b["cost_usd"] = round(b["cost_usd"] + float(r.get("cost_usd", 0.0)), 6)

    bench_by_project: dict[str, dict] = {}
    for r in bench:
        b = bench_by_project.setdefault(r.get("project", "?"), {
            "runs": 0, "measurements": 0, "best_reachable": 0,
            "best_reachable_total": 0, "latest_reachable": 0,
            "reachable_total": 0, "solved_by_runs": 0, "_ever": set()})
        # A benchmark ROW is a measurement, not necessarily a run. `--no-run`
        # writes one, so looking at the scoreboard three times used to read as
        # three runs and inflate every per-run figure derived from the count.
        b["measurements"] += 1
        if r.get("model") and r.get("model") != "(none)":
            b["runs"] += 1
        solved = int(r.get("reachable_solved", 0))
        if solved > b["best_reachable"]:
            b["best_reachable"] = solved
            # Carried WITH the best, because the denominator genuinely varies
            # with disabledEnv between deployments. "latest non-zero wins" meant
            # the ledger could print one run's numerator over another run's
            # total, which is a number that belongs to no run that happened.
            b["best_reachable_total"] = int(r.get("reachable_total", 0))
        b["latest_reachable"] = solved
        b["reachable_total"] = int(r.get("reachable_total", 0)) or b["reachable_total"]
        b["_ever"].update(r.get("solved_keys") or r.get("newly_solved") or [])
        # What the HARNESS did, as opposed to what the board happened to hold.
        # Juice Shop keeps its flags until the container is recreated, so a
        # second run on the same container inherits the first one's score --
        # and `ever_solved` unioned the whole board and presented it as
        # achievement, which can only ever go up.
        b["solved_by_runs"] += int(r.get("solved_this_run",
                                         len(r.get("newly_solved") or [])))
    for b in bench_by_project.values():
        b["ever_solved"] = sorted(b.pop("_ever"))
        b["ever_solved_count"] = len(b["ever_solved"])
        b["best_reachable_total"] = b["best_reachable_total"] or b["reachable_total"]

    out = summarize_runs(runs_dir)
    out.update({
        "missed": sum(int(r.get("missed", 0)) for r in ledger),
        "by_project": by_project, "by_model": by_model, "by_day": by_day,
        "benchmark": bench_by_project,
    })
    return out


# ----------------------------------------------------------------- reading
def read_events(path: str | Path) -> list[LogEvent]:
    p = Path(path)
    if p.is_dir():
        p = p / "events.jsonl"
    out: list[LogEvent] = []
    for line in p.read_text().split("\n"):
        line = line.strip()
        if line:
            out.append(LogEvent(**json.loads(line)))
    return out


_LEVEL_STYLE = {"error": ts.bad, "warn": ts.warn, "info": lambda s: s, "debug": ts.muted}


def render(events: list[LogEvent]) -> str:
    lines = []
    for ev in events:
        clock = ev.ts[11:19] if len(ev.ts) >= 19 else ev.ts
        lvl = _LEVEL_STYLE.get(ev.level, lambda s: s)(f"{ev.level:5}")
        extra = ""
        if ev.event == "policy":
            extra = ts.good("allow") if ev.data.get("allowed") else ts.bad("deny:" + str(ev.data.get("deny_code")))
        elif ev.event == "decision" and ev.data.get("tool"):
            extra = ts.muted(f"{ev.data.get('action')} {ev.data.get('tool')} {ev.data.get('args')}")
        elif ev.event == "tool_result" and ev.data.get("evidence_refs"):
            extra = ts.muted(f"evidence={len(ev.data['evidence_refs'])}")
        elif ev.event == "finding":
            extra = ts.muted(",".join(ev.data.get("cves", [])))
        head = f"{ts.muted(clock)} {lvl} {ts.muted('[' + (ev.actor or '-') + ']')} " \
               f"{ts.name(ev.event)}"
        lines.append(f"{head}  {ev.message} {extra}".rstrip())
    return "\n".join(lines)
