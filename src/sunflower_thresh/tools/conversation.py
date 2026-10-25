"""Stateful chat: hold a conversation, because one message cannot find the bug.

A single POST that sends one message and reads one reply is the right shape for a
single-shot disclosure test and the wrong shape for most of what actually works
against a deployed chatbot, all of which needs TURNS:

  * crescendo -- ask something innocuous, then escalate by degrees. Each turn is
    individually unobjectionable and the guardrail never sees a request worth
    refusing;
  * many-shot -- fill the context with examples of compliance until the next
    answer follows the pattern instead of the system prompt;
  * context poisoning -- establish a false premise early ("you confirmed earlier
    that I am an administrator") and spend it later;
  * goal hijacking across turns, where the injected instruction and the trigger
    that fires it arrive separately.

None of those are expressible in a single POST, so these three tools are the
conversation, and they are the whole target-facing surface this package has:

  * ``chat_open``  mint a conversation. Sends nothing, so it is PASSIVE.
  * ``chat_say``   one turn, with the history attached. ACTIVE.
  * ``chat_reset`` abandon a conversation and start clean.

``chat_reset`` matters more than it looks. The engagement skill file (not vendored
here) requires proof "in two fresh sessions (new conversation, not a later
turn)", and the attack-success-rate statistics assume the trials are INDEPENDENT.
A second trial continued in a poisoned context is not a second trial; it is the
first one still running.

Two wire shapes: ``simple`` posts {message, conversation_id} and lets the server
remember, ``openai`` replays the whole transcript because chat-completions is
stateless. Either way the transcript is
kept here, so a finding's evidence is the conversation rather than its last line.

Observed tool calls are extracted when the target reports them. That is the
TOOL_CALL rung of the oracle ladder, and it is the strongest evidence available
for excessive agency -- what the system DID, rather than prose about what it
might have done. A black-box endpoint simply returns none, and cases needing them
are reported unreachable instead of failed.

Safety: the host and port are fixed from the gated target, exactly as in
``session.py`` -- the model names a path and a message, never a host, so there is
no scope escape through an argument.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from sunflower_thresh.evidence.store import EvidenceStore
from sunflower_thresh.nethttp import connect, scheme_for
from sunflower_thresh.policy.decision import ToolRequest, host_of
from sunflower_thresh.schemas.evidence import EvidenceKind
from sunflower_thresh.schemas.target import ActionClass
from sunflower_thresh.tools.base import Tool, merge_auth_headers

#: Tools whose call sends one message to the target, and therefore spends one unit
#: of the engagement's signed message budget.
#:
#: Named here, beside the tools, rather than in the gate: the gate must not keep a
#: list of which tools are chatty, or it is wrong the first time one is added. The
#: caller reads this and sets `ToolRequest.message_cost`.
MESSAGE_SPENDING_TOOLS: frozenset[str] = frozenset({"chat_say"})


def message_cost_for(tool: str) -> int:
    """How many target messages a call to `tool` spends."""
    return 1 if tool in MESSAGE_SPENDING_TOOLS else 0


class Conversation(BaseModel):
    """One thread with the target, and everything said in it."""

    id: str
    host: str
    port: int = 80
    path: str = "/chat"
    mode: str = "simple"
    message_field: str = "message"
    reply_field: str = "reply"
    #: Where the target reports the tools it called, if it reports them at all.
    tool_calls_field: str = "tool_calls"
    #: Dotted path to the reply when it is not a top-level key:
    #: "data.message.text", "result.0.content". Takes precedence over reply_field.
    #: Customer APIs nest; the harness could only read a flat key.
    reply_path: str = ""
    #: Conversation id the SERVER chose, when it echoes one back. Kept apart from
    #: our own id: ours identifies the thread in the evidence store, theirs is
    #: what the wire needs, and conflating them breaks whichever side we guessed.
    server_session: str = ""
    model: str = ""
    #: None infers from the port (443 and the usual alternates). An explicit value
    #: wins, for an endpoint on a non-standard TLS port.
    tls: bool | None = None
    #: Verified by default. Unverified is a real need on internal targets with
    #: self-signed certificates, and it is never the default.
    verify_tls: bool = True
    extra: dict = Field(default_factory=dict)
    turns: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    opened_at: str = ""
    owner: str = ""

    def messages(self) -> list[dict]:
        """The transcript in chat-completions shape, for ``openai`` mode replay."""
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def sent(self) -> int:
        return sum(1 for t in self.turns if t.get("role") == "user")


class ConversationStore:
    """Live conversations, keyed by (owner, id).

    Keyed by owner for the reason `SessionStore._key` documents at length: one
    slot per target was fine for one agent and is wrong for a fleet. Here the
    stakes are the same shape -- a worker that appends a turn to a peer's
    conversation corrupts both transcripts, and a finding whose evidence is the
    wrong conversation is a false finding with evidence attached.

    Locked for the same reason too: a dict mutated by one worker while another
    reads it hands the reader a half-updated thread.
    """

    def __init__(self) -> None:
        self._convs: dict[tuple[str, str], Conversation] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(cid: str, owner: str) -> tuple[str, str]:
        return (owner or "", cid)

    def open(self, *, host: str, owner: str = "", **cfg) -> Conversation:
        cid = f"conv-{uuid.uuid4().hex[:12]}"
        conv = Conversation(id=cid, host=host, owner=owner or "",
                            opened_at=datetime.now(UTC).isoformat(), **cfg)
        with self._lock:
            self._convs[self._key(cid, owner)] = conv
        return conv

    def get(self, cid: str, *, owner: str = "") -> Conversation | None:
        with self._lock:
            return self._convs.get(self._key(cid, owner))

    def append(self, cid: str, *, owner: str = "", role: str, content: str,
               evidence_ref: str = "", tool_calls=()) -> int:
        """Add a turn. Returns the turn index, or -1 if there is no such thread."""
        with self._lock:
            conv = self._convs.get(self._key(cid, owner))
            if conv is None:
                return -1
            conv.turns.append({"role": role, "content": content,
                               "at": datetime.now(UTC).isoformat(),
                               **({"evidence_ref": evidence_ref} if evidence_ref else {})})
            for call in tool_calls or ():
                conv.tool_calls.append(dict(call))
            return len(conv.turns) - 1

    def close(self, cid: str, *, owner: str = "") -> None:
        with self._lock:
            self._convs.pop(self._key(cid, owner), None)

    def transcript(self, cid: str, *, owner: str = "") -> list[dict]:
        conv = self.get(cid, owner=owner)
        return list(conv.turns) if conv else []

    def messages_sent(self, *, owner: str = "") -> int:
        """User turns sent, for an owner or across every conversation.

        The gate counts target messages itself -- that is the budget that must
        hold across workers. This is the same figure from the tool's side, for
        the orchestrator's prompt and for a per-conversation cap.
        """
        with self._lock:
            return sum(c.sent() for (own, _cid), c in self._convs.items()
                       if not owner or own == owner)

    def live(self, *, owner: str = "") -> list[str]:
        with self._lock:
            return [cid for (own, cid) in self._convs if not owner or own == owner]


class _ChatTool(Tool):
    """Shared plumbing: every verb needs the store and the same arg handling."""

    def __init__(self, conversations: ConversationStore,
                 store: EvidenceStore | None = None, policy=None) -> None:
        super().__init__(store)
        self.conversations = conversations
        #: The gate, read at CALL time for this project's signed auth headers. This
        #: is the pattern `PolicyEngine.inventory_for` documents -- "exposed so a
        #: tool can read a capability switch rather than hardcoding a policy of its
        #: own" -- and the reason the headers live on the inventory rather than in a
        #: tool argument: they are the authority the engagement acts under, they are
        #: signed with the scope, and the model must never be able to set one.
        self.policy = policy

    @staticmethod
    def _owner(req: ToolRequest) -> str:
        return req.requested_by or ""

    def _engagement_headers(self, req: ToolRequest) -> dict:
        """This project's resolved auth headers, or nothing."""
        policy = self.policy
        if policy is None:
            return {}
        try:
            inv = policy.inventory_for(req.project)
        except Exception:  # noqa: BLE001 - a header lookup never breaks a probe
            return {}
        if inv is None:
            return {}
        try:
            return inv.resolved_auth_headers()
        except Exception:  # noqa: BLE001
            return {}


class ChatOpen(_ChatTool):
    """Mint a conversation. PASSIVE: nothing is sent to the target."""

    name = "chat_open"
    action_class = ActionClass.PASSIVE
    evidence_kind = EvidenceKind.LOG

    def __call__(self, req: ToolRequest) -> dict:
        host = host_of(req)
        if not host:
            return {"error": "chat_open requires a target host (ip or domain)",
                    "evidence_refs": []}
        a = req.args or {}
        conv = self.conversations.open(
            host=host, owner=self._owner(req),
            port=int(a.get("port", 80)), path=a.get("path", "/chat"),
            mode=a.get("mode", "simple"),
            message_field=a.get("message_field", "message"),
            reply_field=a.get("reply_field", "reply"),
            tool_calls_field=a.get("tool_calls_field", "tool_calls"),
            reply_path=a.get("reply_path", ""),
            model=a.get("model", ""), extra=a.get("extra", {}) or {},
            tls=a.get("tls"), verify_tls=bool(a.get("verify_tls", True)))
        return {"conversation_id": conv.id, "target": conv.host, "port": conv.port,
                "path": conv.path, "mode": conv.mode, "turns": 0,
                "evidence_refs": []}


class ChatReset(_ChatTool):
    """Abandon a conversation. PASSIVE, and the independence guarantee for ASR."""

    name = "chat_reset"
    action_class = ActionClass.PASSIVE
    evidence_kind = EvidenceKind.LOG

    def __call__(self, req: ToolRequest) -> dict:
        cid = (req.args or {}).get("conversation_id", "")
        if not cid:
            return {"error": "chat_reset requires a conversation_id", "evidence_refs": []}
        owner = self._owner(req)
        conv = self.conversations.get(cid, owner=owner)
        turns = len(conv.turns) if conv else 0
        self.conversations.close(cid, owner=owner)
        return {"conversation_id": cid, "closed": conv is not None,
                "turns_discarded": turns, "evidence_refs": []}


class ChatSay(_ChatTool):
    """One turn in a conversation, with the history attached.

    The reply, the tool calls the target admits to, and the whole transcript so
    far are returned -- the oracle ladder reads all three, and a crescendo's proof
    is the conversation rather than its last line.
    """

    name = "chat_say"
    action_class = ActionClass.ACTIVE
    evidence_kind = EvidenceKind.HTTP_EXCHANGE

    def __call__(self, req: ToolRequest) -> dict:
        a = req.args or {}
        message = a.get("message")
        if not message:
            return {"error": "chat_say requires a message", "evidence_refs": []}
        owner = self._owner(req)
        cid = a.get("conversation_id", "")
        conv = self.conversations.get(cid, owner=owner) if cid else None
        if conv is None:
            # A turn with no thread opens one, so a single-turn case does not need
            # two calls. The model forgetting chat_open should cost it nothing.
            host = host_of(req)
            if not host:
                return {"error": "chat_say requires a target host or a conversation_id",
                        "evidence_refs": []}
            conv = self.conversations.open(
                host=host, owner=owner, port=int(a.get("port", 80)),
                path=a.get("path", "/chat"), mode=a.get("mode", "simple"),
                message_field=a.get("message_field", "message"),
                reply_field=a.get("reply_field", "reply"),
                tool_calls_field=a.get("tool_calls_field", "tool_calls"),
                reply_path=a.get("reply_path", ""),
                model=a.get("model", ""), extra=a.get("extra", {}) or {},
                tls=a.get("tls"), verify_tls=bool(a.get("verify_tls", True)))
            cid = conv.id

        # The gate pinned the host; an argument cannot move it. Compared against
        # whichever identifier the conversation was opened with, so a hostname
        # engagement is not rejected by a check written for addresses.
        asked = host_of(req)
        if asked and asked != conv.host:
            return {"error": f"conversation {cid} belongs to {conv.host}, "
                             f"not {asked}", "evidence_refs": []}

        self.conversations.append(cid, owner=owner, role="user", content=message)
        conv = self.conversations.get(cid, owner=owner)
        body = self._body(conv, message)
        started = datetime.now(UTC)
        # Engagement headers UNDER the caller's, using the same precedence rule
        # session.merge_auth_headers already enforces and documents: an explicit
        # header is the caller saying what it wants, and silently overriding it is
        # the bug that made a forged-JWT test unable to pass or fail.
        headers = merge_auth_headers(a.get("headers") or {},
                                     self._engagement_headers(req))
        try:
            status, reason, raw = self._post(conv, body,
                                             float(a.get("timeout_s", 60)),
                                             headers=headers)
        except OSError as exc:
            return {"error": f"chat_say transport failed: {exc}",
                    "conversation_id": cid, "evidence_refs": []}
        latency = (datetime.now(UTC) - started).total_seconds()
        reply, tool_calls, server_id, usage = self._parse(conv, raw)
        if server_id and not conv.server_session:
            conv.server_session = server_id

        exchange = (f"POST {scheme_for(conv.port, conv.tls)}://{conv.host}:{conv.port}"
                    f"{conv.path}\nHost: {conv.host}:{conv.port}\n\n"
                    f"{json.dumps(body)}\n\nHTTP/1.1 {status} {reason}\n\n{raw[:4000]}")
        ref = self._record(exchange.encode(), target=conv.host,
                           meta={"path": conv.path, "mode": conv.mode,
                                 "conversation_id": cid, "turn": conv.sent()})

        if status >= 400:
            # An HTTP error body is NOT a reply, and nothing downstream could tell.
            # `_parse` falls back to the raw body when it finds no reply field, so a
            # 422 that echoes the request -- a FastAPI validation error, a gateway's
            # error document -- was handed to the oracles as the assistant's answer.
            # The harness's own attack text is IN that echo, so the structural and
            # callback rungs matched our own payload.
            #
            # Measured against a real server answering 422 to everything, with the
            # SHIPPED `chat-html-injection` case: "VALIDATED: ASR 1.00 (10/10, 95% CI
            # 0.72-1.00), matched control 0/10, p=0.000. Evidence: STRUCTURAL --
            # regex matched '<script'." The arms separate because the two turns
            # differ in text and the echo reproduces whichever was sent, so the
            # attack arm always matches itself and the control never does. A
            # significant p-value against an endpoint that never answered once.
            #
            # `error` is the channel `_ChatRunner.trial` already reads to discard a
            # trial, and the exchange is recorded ABOVE, so the failure stays
            # provable rather than merely absent. The turn is not appended to the
            # transcript: a conversation must not carry an error document as a thing
            # the assistant said.
            return {"error": f"chat_say got HTTP {status} {reason}".strip(),
                    "http_status": status, "conversation_id": cid,
                    "target": conv.host, "port": conv.port, "path": conv.path,
                    "status": status, "body_excerpt": raw[:400],
                    "latency_s": round(latency, 3),
                    "at": datetime.now(UTC).isoformat(),
                    "evidence_refs": [ref]}

        self.conversations.append(cid, owner=owner, role="assistant", content=reply,
                                  evidence_ref=ref, tool_calls=tool_calls)
        conv = self.conversations.get(cid, owner=owner)
        return {"conversation_id": cid, "target": conv.host, "port": conv.port,
                "path": conv.path, "status": status, "reply": reply,
                "tool_calls": tool_calls, "usage": usage,
                "latency_s": round(latency, 3), "turns": len(conv.turns),
                "transcript": conv.messages(),
                "at": datetime.now(UTC).isoformat(),
                "evidence_refs": [ref]}

    # ----- wire -----

    @staticmethod
    def _body(conv: Conversation, message: str) -> dict:
        if conv.mode == "openai":
            # Chat-completions is stateless, so the HISTORY is the state. Replaying
            # it is what makes a crescendo work against an OpenAI-compatible
            # endpoint; sending only the last turn would restart the attack every
            # time and read as the guardrail holding.
            return {"model": conv.model, "messages": conv.messages(), **conv.extra}
        body = {conv.message_field: message, **conv.extra}
        # Server-side memory: hand back whichever id the server recognises, from
        # the FIRST turn onwards.
        #
        # This withheld the id until turn two, on the reasoning that a brand-new
        # thread has nothing to resume. That was wrong: a server with no id in
        # hand opens a fresh session, so turn one landed in a session of its own
        # and the thread only began accumulating from turn two. A four-turn
        # crescendo then reached the target as turns 1,2,3 -- one short -- and the
        # attack read as the guardrail holding.
        body["conversation_id"] = conv.server_session or conv.id
        return body

    @staticmethod
    def _post(conv: Conversation, body: dict, timeout: float, headers: dict | None = None):
        path = conv.path
        if conv.mode == "openai" and path == "/chat":
            path = "/v1/chat/completions"
        # TLS where the port or the case says so. Every real chat endpoint is
        # HTTPS; the loopback lab is why plain HTTP survived this long.
        http_conn = connect(conv.host, conv.port, tls=conv.tls, timeout=timeout,
                            verify=conv.verify_tls)
        try:
            payload = json.dumps(body).encode()
            http_conn.request("POST", path, body=payload,
                              headers={"Host": conv.host,
                                       "Content-Type": "application/json",
                                       "User-Agent": "harness-chat/0.2",
                                       **(headers or {})})
            resp = http_conn.getresponse()
            return resp.status, resp.reason, resp.read(65536).decode(errors="replace")
        finally:
            http_conn.close()

    @staticmethod
    def _parse(conv: Conversation, raw: str):
        """(reply, tool_calls, server_conversation_id, usage) from a response body."""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Not one JSON document. Most likely a stream: Server-Sent Events or
            # newline-delimited JSON, which is how a great many production chat
            # endpoints answer and which this used to hand back RAW. That is not a
            # cosmetic problem -- the whole SSE framing ("data: {...}" repeated, plus
            # "[DONE]") became the "reply", so a canary oracle matched on protocol
            # scaffolding and tool calls were lost entirely.
            streamed = _from_stream(raw)
            if streamed is not None:
                return streamed
            return raw[:4000], [], "", {}
        if not isinstance(obj, dict):
            return raw[:4000], [], "", {}
        usage = obj.get("usage") or {}
        server_id = str(obj.get("conversation_id") or obj.get("session_id") or "")
        if conv.mode == "openai":
            try:
                msg = obj["choices"][0]["message"]
                return (msg.get("content") or "",
                        _as_calls(msg.get("tool_calls")), server_id, usage)
            except (KeyError, IndexError, TypeError):
                return raw[:4000], [], server_id, usage
        reply = _dig(obj, conv.reply_path) if conv.reply_path else obj.get(conv.reply_field)
        if reply is None:
            # The SAME ladder `_from_stream` walks, applied to ONE document. These
            # two extractors disagreed: the stream path reads eight real provider
            # shapes, and this path read one key and then handed back the whole
            # JSON envelope. So identical bytes measured two different things
            # depending only on whether the target streamed them.
            #
            # That is a measurement bug, not a tidy-up. The string goes unmodified
            # into the judge's rubric under "--- ASSISTANT REPLY ---", into
            # `ResponseLengthOracle`'s character count, and into every regex rung.
            # Against a target answering `{"text": "I'm sorry, I can't share
            # that."}` the recorded reply was 42 characters of JSON rather than 30
            # of prose, and `^I'm sorry` did not match -- a refusal scored as a
            # non-refusal, in the direction that invents findings.
            #
            # `_stream_text` returns "" rather than None when nothing matches, so
            # `or raw[:4000]` keeps the deliberately forgiving fallback `_dig`'s
            # docstring promises. An explicit `reply_path` and a configured
            # `reply_field` are still tried FIRST, so this is a fallback and never
            # an override.
            reply = _stream_text(obj) or raw[:4000]
        return (str(reply), _as_calls(obj.get(conv.tool_calls_field)), server_id, usage)


def _dig(obj, path: str):
    """Walk a dotted path, tolerating list indices. None if it does not resolve.

    Deliberately forgiving: a wrong `reply_path` must degrade to "no reply found"
    and let the caller fall back to the raw body, not raise mid-run. Preflight is
    where a wrong path should be caught, by printing what was extracted.
    """
    cur = obj
    for part in (path or "").split("."):
        if part == "":
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


#: Frames a stream ends with, which are not part of the reply.
_STREAM_SENTINELS = frozenset({"[DONE]", "DONE", ""})


def _from_stream(raw: str):
    """(reply, tool_calls, server_id, usage) from SSE or newline-delimited JSON.

    Concatenates the text deltas in order and merges the tool calls, because a
    streamed answer arrives as dozens of fragments and the reply is their
    concatenation -- not the last one, and certainly not the framing around them.
    Returns None when this does not look like a stream, so the caller can fall back.
    """
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    payloads: list[str] = []
    for line in lines:
        if line.startswith("data:"):
            payloads.append(line[5:].strip())
        elif line.startswith(("event:", "id:", "retry:", ":")):
            continue            # SSE metadata, not content
        elif line.startswith("{"):
            payloads.append(line)       # newline-delimited JSON
        else:
            return None
    payloads = [p for p in payloads if p not in _STREAM_SENTINELS]
    if not payloads:
        return None
    parts: list[str] = []
    calls: list[dict] = []
    server_id = ""
    usage: dict = {}
    for payload in payloads:
        try:
            frame = json.loads(payload)
        except json.JSONDecodeError:
            parts.append(payload)       # a plain-text delta
            continue
        if not isinstance(frame, dict):
            parts.append(str(frame))
            continue
        usage = frame.get("usage") or usage
        server_id = str(frame.get("conversation_id") or frame.get("id") or server_id)
        text = _stream_text(frame)
        if text:
            parts.append(text)
        calls.extend(_as_calls(_stream_calls(frame)))
    return ("".join(parts)[:8000], calls, server_id, usage)


def _stream_text(frame: dict) -> str:
    """The text delta in one frame, across the shapes providers actually send."""
    for path in ("choices.0.delta.content", "choices.0.message.content",
                 "delta.text", "delta.content", "content", "text", "reply",
                 "message.content"):
        got = _dig(frame, path)
        if isinstance(got, str) and got:
            return got
    return ""


def _stream_calls(frame: dict):
    for path in ("choices.0.delta.tool_calls", "choices.0.message.tool_calls",
                 "tool_calls"):
        got = _dig(frame, path)
        if got:
            return got
    return []


def _as_calls(value) -> list[dict]:
    """Normalise whatever shape the target reports tool calls in.

    OpenAI nests them under ``function``; a custom bot usually does not. The
    oracle asks one question -- which tool, with which arguments -- so both
    shapes are flattened to that rather than taught to the oracle.
    """
    out: list[dict] = []
    for call in value or []:
        if not isinstance(call, dict):
            out.append({"name": str(call), "arguments": ""})
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") or call.get("name") or call.get("tool") or ""
        args = fn.get("arguments", call.get("arguments", call.get("args", "")))
        out.append({"name": str(name),
                    "arguments": args if isinstance(args, str) else json.dumps(args,
                                                                              sort_keys=True)})
    return out


#: Every tool name this package can actually serve.
#:
#: Declared beside the factory below and asserted against its keys by a test, so
#: the two cannot disagree. It is the fact `preflight.KNOWN_TOOLS` is built from.
#:
#: Why it is a declared fact rather than a list someone keeps: `tool_allowed` is
#: exact membership, so a permitted name with nothing behind it is not a typo in a
#: permission, it is no permission at all. Preflight already catches a name
#: OUTSIDE this set, and its own docstring says a denial it predicts "cannot
#: surprise an operator mid-window". A name INSIDE the set with no implementation
#: went the other way and was worse: the gate ALLOWED it, `execute_tool` returned
#: "no implementation registered", the message was refunded, and preflight had
#: printed the name in its PASS line as though granted. `chat_probe` was in that
#: state -- a complete transport in `tools/chat.py` that no driver ever built.
SERVABLE_TOOLS: frozenset[str] = frozenset({"chat_open", "chat_say", "chat_reset"})


def chat_tools(conversations: ConversationStore, store: EvidenceStore,
               *, policy=None) -> dict[str, Tool]:
    """The tool table a driver registers with `AgentContext`.

    One factory because there were three byte-identical copies of this dict --
    `demo.py`, `calibrate.py` and `tests/test_whole_chain.py` -- and a fourth
    driver would have made a fourth. Three copies of a registry is three chances
    to add a tool to two of them, and the symptom is a tool that works in the demo
    and is missing under calibration.
    """
    return {t.name: t for t in (ChatOpen(conversations, store, policy=policy),
                                ChatSay(conversations, store, policy=policy),
                                ChatReset(conversations, store, policy=policy))}
