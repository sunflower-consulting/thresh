"""One model call, over the standard library, for the roles that are not the target.

stem talks to exactly one kind of thing by default: the application under test,
through `tools/conversation.py`, gated and budgeted. The judge is not that. It
never touches the target, proposes nothing and holds no tools -- it reads a reply
and returns a verdict -- so it needs its own way out to its own endpoint.

WHY THIS IS NOT A MODEL CATALOG.

The engine this was extracted from has `models/base.py`, `models/registry.py`,
`models/providers/{openai_compat,anthropic,mock,scripted}.py` and a YAML catalog,
so swapping a model is editing a config file. That is the right shape for a tool
with several model roles and a cost ledger. stem has one, named by one pair of
environment variables, and a registry with a single row is a schema to maintain
for nothing. If a second role ever lands -- an attacker, say -- reintroduce the
catalog rather than growing a second special case here.

WHAT IT KEEPS, BECAUSE EACH OF THESE WAS PAID FOR ONCE ALREADY.

`_REASONING_KEYS`: a reasoning model may put its whole answer in `reasoning` with
`content` empty. gpt-oss does. Without the fallback every call against it reads as
an empty reply, and for a judge that is indistinguishable from "the judge did not
answer" -- which is exactly the state that silently disables a suppressor.

`ModelReply.truncated`: read from the raw response, because it is the only
authoritative signal that the model ran out of budget mid-answer. It lives here
rather than in `judge.py` because it is a fact about a response shape, and the
transport owns response shapes. What it is FOR is in `judge.py`, where refusing
the loose parse on a truncated reply is the difference between "ask again with
more budget" and "lower the threshold".

HTTP 429 is excepted from the no-retry-on-4xx rule: it is the one 4xx that means
retry. And a plain 500 is retried because Ollama intermittently returns one from
its own template on its own output.

WHAT IT DELIBERATELY DOES NOT KEEP, and when to put it back. The engine bounds
one request in flight per ENDPOINT, because a local server serves one request at a
time and two callers against it get one caller's throughput plus the queueing.
stem's sweep is single-threaded, so there is nothing to contend. The moment any
caller here runs arms concurrently against one local endpoint, that bound is
required rather than nice.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from sunflower_stem.nethttp import connect

#: Cap generation unless a caller says otherwise. Not 2048: the engine's catalog
#: records 512 producing `finish_reason: length` on every call to a reasoning
#: model, because it spends output budget thinking before it answers. An uncapped
#: local model can ramble until the timeout instead, so this is raised rather than
#: removed.
DEFAULT_MAX_TOKENS = 8192

#: A local model on a 16 GB card may be loading from disk when the first call
#: arrives, and a measured cold load on the reference box is 9 to 19 seconds.
DEFAULT_TIMEOUT_S = 120.0

#: One retry is not enough against a server that 500s on its own template.
DEFAULT_RETRIES = 2

#: Where a reasoning model may put its answer when `content` comes back empty.
_REASONING_KEYS = ("reasoning", "reasoning_content", "thinking")

#: Parameters that an OpenAI-compatible server ACCEPTS AND IGNORES, which is worse
#: than rejecting them.
#:
#: All of the below is measured against Ollama 0.40.5 with qwen3:8b, one prompt.
#: Across the two endpoints it serves:
#:
#:     /v1/chat/completions   baseline      327 completion tokens, 1387 chars reasoning
#:     /v1/chat/completions   think=False   253 completion tokens, 1012 chars reasoning
#:     /api/chat              baseline      439 eval tokens,       1857 chars thinking
#:     /api/chat              think=False    13 eval tokens,          0 chars thinking
#:
#: So the lever is real and worth roughly 34x the generated tokens, and it does
#: nothing on the endpoint this transport speaks. A caller passing it would get a
#: thinking model, the token bill, and the belief they had turned thinking off.
#:
#: That belief is expensive here specifically. Thinking is what makes a judge run
#: out of output budget before it reaches a verdict, which is the exact condition
#: `ModelReply.truncated` exists to catch -- so "I disabled thinking" and "my judge
#: keeps truncating" would be held at the same time, and the second blamed on
#: max_tokens.
#:
#: TWO FIXES WERE TRIED AND NEITHER WORKS, which is why the message below does not
#: offer one:
#:
#:   - `PARAMETER think false` in a Modelfile: `ollama create` fails outright with
#:     "unknown parameter 'think'". It is a request option, not a model parameter.
#:   - Qwen3's `/no_think` soft switch through this endpoint: appended to the user
#:     turn it measured 410 completion tokens against a 357 baseline, and in a
#:     system turn 210, with 849 to 1718 chars of reasoning throughout. Within
#:     noise, and in the wrong direction half the time.
#:
#: So on this endpoint, for this model, thinking cannot be switched off. The
#: practical answer is to budget for it rather than defeat it -- which
#: `DEFAULT_MAX_TOKENS` at 8192 already does, against a measured 1522-char
#: reasoning blob -- or to choose a judge model that does not think.
#:
#: Refused rather than warned, because the failure is a silent belief and an error
#: that carries the measurement is how a caller learns it in seconds.
#:
#: NAMED FOR THE SERVER IT WAS MEASURED ON, which it was not. This was
#: `IGNORED_BY_OPENAI_SHAPE` and applied to every OpenAI-compatible endpoint, which
#: turned a measurement about one server into a refusal for all of them --
#: `reasoning_effort` is a real OpenAI parameter, as the entry below says in its own
#: words, and `chat_template_kwargs` is supported by vLLM and others. So a valid
#: request to one of those was rejected before it ever reached the server, on the
#: strength of what Ollama does. Caught in review.
#:
#: Refusing something a server does not implement is not fail-closed. It is a false
#: claim about that server's capability, so the guard now applies only where the
#: measurement does and anything unrecognised is passed through to be answered by
#: the thing that knows.
def _looks_like_ollama(endpoint: str) -> bool:
    """A best guess from the URL, which is all there is without sending a request.

    Ollama's default port and the obvious hostnames. Wrong in both directions for
    an unusual deployment, which is why `backend=` overrides it -- and wrong in the
    safe direction by default: an unrecognised endpoint is not refused anything.
    """
    low = (endpoint or "").lower()
    return ":11434" in low or "ollama" in low


IGNORED_BY_OLLAMA = {
    "think": ("Ollama accepts `think` on /v1/chat/completions and ignores it: "
              "measured 253 completion tokens against a 327 baseline, still 1012 "
              "chars of reasoning. It works only on the native /api/chat (13 "
              "tokens, no thinking), it is NOT a Modelfile parameter (`ollama "
              "create` rejects it), and Qwen3's /no_think switch is not honoured "
              "here either. Budget for thinking instead, or pick a judge model "
              "that does not think."),
    "chat_template_kwargs": (
        "accepted and ignored on this endpoint: measured 275 completion tokens "
        "against a 327 baseline, still 1082 chars of reasoning. Ollama does not "
        "pass it to the template."),
    "reasoning_effort": (
        "an OpenAI-API parameter Ollama does not implement; measured "
        "indistinguishable from the baseline. There is no endpoint-side lever for "
        "this model."),
}


#: Enough for a verbose reasoning reply, bounded so a misbehaving endpoint cannot
#: exhaust memory. Sized against a measured 1522-char reasoning blob with room to
#: spare, since thinking cannot be switched off on this endpoint.
_MAX_RESPONSE_BYTES = 1_048_576


class ModelCallError(RuntimeError):
    """An endpoint was unreachable or answered with an error."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ModelReply:
    """What came back, plus the one thing about it that changes how it is read."""

    model_id: str
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True when the model hit its output cap, so what it sent is unfinished.

        OpenAI-shaped servers say `finish_reason: "length"`; Anthropic-shaped ones
        say `stop_reason: "max_tokens"`. Both are checked because the endpoint is
        whatever the operator pointed at.
        """
        raw = self.raw
        if not isinstance(raw, dict):
            return False
        try:
            choice = (raw.get("choices") or [{}])[0] or {}
        except (IndexError, AttributeError, TypeError):
            choice = {}
        if not isinstance(choice, dict):
            choice = {}
        if str(choice.get("finish_reason") or "").lower() == "length":
            return True
        return str(raw.get("stop_reason") or "").lower() == "max_tokens"


def _answer(message: dict) -> str:
    """The assistant's text, falling back to a reasoning field if content is empty.

    Preferring `content` leaves ordinary models untouched; the fallback only fires
    when there would otherwise be nothing at all to parse.
    """
    text = message.get("content") or ""
    if str(text).strip():
        return str(text)
    for key in _REASONING_KEYS:
        alt = message.get(key)
        if alt and str(alt).strip():
            return str(alt)
    return str(text)


class ChatTransport:
    """An OpenAI-compatible chat endpoint, reached with `nethttp.connect`.

    `nethttp` rather than `urllib.request` on purpose: it is the one place in the
    package that opens a connection, it decides http against https by port, and it
    verifies TLS by default while handing a caller back the fact when it does not.
    The bundled lab's own Ollama caller predates it and should not be copied.
    """

    def __init__(self, endpoint: str, model_name: str, *, api_key_env: str = "",
                 timeout_s: float = DEFAULT_TIMEOUT_S, retries: int = DEFAULT_RETRIES,
                 max_tokens: int = DEFAULT_MAX_TOKENS, verify_tls: bool = True,
                 defaults: dict | None = None,
                 backend: str = "auto") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.max_tokens = int(max_tokens)
        self.verify_tls = bool(verify_tls)
        self.defaults = dict(defaults or {})
        #: Whether the refusals above apply. Detected from the endpoint and
        #: overridable, because the measurements behind them are about Ollama and
        #: a parameter another server implements must reach it.
        self.is_ollama = (_looks_like_ollama(self.endpoint) if backend == "auto"
                          else backend == "ollama")

    @property
    def model_id(self) -> str:
        return self.model_name

    def complete(self, messages: list[Message], **params) -> ModelReply:
        merged = {**self.defaults, **params}
        if self.is_ollama:
            for name in sorted(set(merged) & set(IGNORED_BY_OLLAMA)):
                raise ModelCallError(
                    f"{name!r} does nothing on Ollama's OpenAI-compatible "
                    f"endpoint. {IGNORED_BY_OLLAMA[name]}")
        # A falsy cap means "no cap, let the server decide": an escape hatch, not
        # the default.
        cap = merged.pop("max_tokens", self.max_tokens)
        if cap:
            merged["max_tokens"] = int(cap)
        body = {"model": self.model_name,
                "messages": [m.as_dict() for m in messages],
                **merged}
        raw = self._send("POST", "/chat/completions", body)
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelCallError(
                f"unexpected response shape from {self.model_name!r}: "
                f"{str(raw)[:300]}") from exc
        usage_raw = raw.get("usage") or {}
        usage = TokenUsage(prompt_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
                           completion_tokens=int(usage_raw.get("completion_tokens", 0) or 0))
        return ModelReply(model_id=self.model_name, text=_answer(message),
                          usage=usage, raw=raw)

    def probe(self) -> dict:
        """Reachability, without spending completion tokens. Never raises.

        A caveat worth knowing before trusting it: a `GET /models` listing is what
        succeeds while a server is up but cannot actually run a particular model.
        On the reference box a non-tray Ollama launch serves small models and
        returns HTTP 500 on the large one, and this probe reports that server
        healthy. So this answers "is anything there", and only a completion
        against the model you need answers "can it run".
        """
        start = time.monotonic()
        try:
            raw = self._send("GET", "/models", None)
        except Exception as exc:  # noqa: BLE001 - a probe never raises
            return {"ok": False, "endpoint": self.endpoint,
                    "error": f"{type(exc).__name__}: {exc}"[:200]}
        served = [m.get("id") for m in (raw.get("data") or [])] if isinstance(raw, dict) else []
        return {"ok": True, "endpoint": self.endpoint, "model_name": self.model_name,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "served_models": [s for s in served if s][:20]}

    # ----------------------------------------------------------------- helpers
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "stem-judge/0.1"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _send(self, method: str, path: str, body: dict | None) -> dict:
        split = urlsplit(self.endpoint if "//" in self.endpoint
                         else f"http://{self.endpoint}")
        host = split.hostname or "127.0.0.1"
        tls = split.scheme == "https"
        port = split.port or (443 if tls else 80)
        full = f"{split.path.rstrip('/')}{path}"
        payload = json.dumps(body).encode() if body is not None else None

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._once(host, port, tls, method, full, payload)
            except ModelCallError as exc:
                # A 4xx is deterministic and not worth repeating, except 429,
                # which is the one 4xx that means retry.
                if getattr(exc, "fatal", False):
                    raise
                last = exc
            if attempt < self.retries:
                time.sleep(0.4 * (attempt + 1))
        assert last is not None
        raise last

    def _once(self, host, port, tls, method, path, payload) -> dict:
        conn = connect(host, port, tls=tls, timeout=self.timeout_s,
                       verify=self.verify_tls)
        try:
            conn.request(method, path, body=payload, headers=self._headers())
            resp = conn.getresponse()
            text = resp.read(_MAX_RESPONSE_BYTES).decode(errors="replace")
            if resp.status >= 400:
                err = ModelCallError(
                    f"{self.model_name!r} HTTP {resp.status} at {self.endpoint}: "
                    f"{text[:300]}")
                err.fatal = 400 <= resp.status < 500 and resp.status != 429
                raise err
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ModelCallError(
                    f"{self.model_name!r} returned non-JSON: {text[:200]}") from exc
        except (OSError, TimeoutError) as exc:
            raise ModelCallError(
                f"{self.model_name!r} unreachable at {self.endpoint}: {exc}") from exc
        finally:
            conn.close()


class ScriptedTransport:
    """Fixed replies, for exercising a caller with no endpoint and no GPU.

    This is what the dry calibration pass runs on. A judge that always says yes
    must be reported too eager and one that always says no too timid, and proving
    the read-out can tell those apart is what makes a live run worth the GPU time.
    """

    def __init__(self, model_id: str, replies: list[str] | str,
                 *, finish_reason: str = "stop") -> None:
        self._model_id = model_id
        self._replies = [replies] if isinstance(replies, str) else list(replies)
        self._finish_reason = finish_reason
        self.calls: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, messages: list[Message], **_params) -> ModelReply:
        self.calls.append(list(messages))
        if not self._replies:
            raise ModelCallError(f"{self._model_id!r} has no scripted replies left")
        text = self._replies[0] if len(self._replies) == 1 else self._replies.pop(0)
        raw = {"choices": [{"message": {"content": text},
                            "finish_reason": self._finish_reason}]}
        return ModelReply(model_id=self._model_id, text=text, raw=raw)

    def probe(self) -> dict:
        return {"ok": True, "endpoint": "scripted", "model_name": self._model_id,
                "served_models": [self._model_id]}
