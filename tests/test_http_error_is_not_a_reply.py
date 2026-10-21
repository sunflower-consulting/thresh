"""An HTTP error body is not something the assistant said.

`ChatSay.__call__` read `status` only to build the evidence string. It never
gated on it, returned no `error` key for any HTTP code, and `_parse` falls back
to the raw body when it finds no reply field -- so a 4xx/5xx body was appended to
the transcript as the assistant's turn, rendered by `VictimRenderer`, and scored
by the ladder.

Measured against a real server, no stubbing, with the SHIPPED
`chat-html-injection` case and a target answering 422 to every POST:

    attack 10/10  control 0/10  asr=1.0  p=5e-06  truncated=False
    VALIDATED: ASR 1.00 (10/10, 95% CI 0.72-1.00), matched control 0/10,
    p=0.000. Evidence: STRUCTURAL -- regex matched '<script'.

The arms separate because the two turns differ in text and the echo reproduces
whichever was sent, so the attack arm always matches its own payload and the
control arm never does. A significant p-value, a VALIDATED finding and a
customer-facing number, from an endpoint that never answered once.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sunflower_thresh.evidence.store import EvidenceStore
from sunflower_thresh.policy.decision import Decision
from sunflower_thresh.tools.chat import BUILTIN_CHAT_CASES
from sunflower_thresh.tools.conversation import ConversationStore, chat_tools
from sunflower_thresh.validation_chat import validate_chat_case


def _echo_server(code: int):
    """A server that answers `code` and echoes the request body, as a validation
    error document does. The echo is the point: it is what put our own payload
    where the oracles could match it."""
    posts: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            posts.append(body)
            out = json.dumps({"detail": [{"loc": ["body"], "msg": "invalid",
                                          "input": json.loads(body or b"{}")}]}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, posts


class _Ctx:
    """The real chat tools, with the gate allowing everything. The defect was in
    the transport, so stubbing the transport would hide it."""

    project = "http-error"

    def __init__(self):
        self._impl = chat_tools(ConversationStore(), EvidenceStore())

    def execute_tool(self, req):
        return (Decision(allowed=True, reason="ok", request=req),
                self._impl[req.tool](req))


@pytest.mark.parametrize("code", [400, 401, 422, 500, 503])
def test_an_http_error_never_becomes_a_measured_result(code):
    """No HTTP error code may produce a finding, whatever the body says."""
    srv, posts = _echo_server(code)
    try:
        case = BUILTIN_CHAT_CASES["chat-html-injection"].model_copy(deep=True)
        case.trials = 10
        case.control_trials = 10

        finding, _ = validate_chat_case(
            _Ctx(), case, target_ip="127.0.0.1", port=srv.server_address[1],
            run_id=f"err-{code}")

        t = finding.trials
        assert t.hits == 0 and t.trials == 0, (
            f"HTTP {code} produced {t.hits}/{t.trials} measured trials; the error "
            f"body was scored as the assistant's reply")
        assert finding.state.value == "under_test", (
            f"HTTP {code} reached state {finding.state.value} -- "
            f"{finding.validation_result[:120]}")
        assert t.truncated is True
        assert "UNTESTED" in finding.validation_result

        # And it must stop, rather than spending the budget on a dead endpoint.
        assert len(posts) == 1, (
            f"{len(posts)} POSTs sent to an endpoint answering {code}; one is "
            f"enough to know")
    finally:
        srv.shutdown()


def test_a_200_from_the_same_echo_server_does_produce_a_finding():
    """The control, and it is what makes the test above mean anything.

    The same echoing server, the same case, the same harness -- only the status
    changes. If this did not produce a VALIDATED finding then the assertions
    above would pass for a harness that simply never measures anything, and the
    false positive they exist to catch would be indistinguishable from the fix.
    """
    srv, posts = _echo_server(200)
    try:
        case = BUILTIN_CHAT_CASES["chat-html-injection"].model_copy(deep=True)
        case.trials = 5
        case.control_trials = 5

        finding, _ = validate_chat_case(
            _Ctx(), case, target_ip="127.0.0.1", port=srv.server_address[1],
            run_id="err-200")

        t = finding.trials
        assert (t.hits, t.trials) == (5, 5), (
            f"the echo at HTTP 200 measured {t.hits}/{t.trials}; this control must "
            f"reach a finding or the test above proves nothing")
        assert finding.state.value == "validated"
        assert len(posts) == 10
    finally:
        srv.shutdown()


def test_the_failed_exchange_is_still_recorded_as_evidence():
    """A refused trial must stay provable. The HTTP exchange is recorded BEFORE
    the error return, so "the target answered 503" is answerable from the
    evidence store rather than only from an absence."""
    srv, _ = _echo_server(503)
    try:
        store = EvidenceStore()
        impl = chat_tools(ConversationStore(), store)

        class Ctx:
            project = "http-error"

            def execute_tool(self, req):
                return (Decision(allowed=True, reason="ok", request=req),
                        impl[req.tool](req))

        from sunflower_thresh.policy.decision import ToolRequest, target_ref
        from sunflower_thresh.schemas.target import ActionClass

        port = srv.server_address[1]
        ctx = Ctx()
        _, opened = ctx.execute_tool(ToolRequest(
            project="http-error", tool="chat_open", action_class=ActionClass.PASSIVE,
            target=target_ref("127.0.0.1"), requested_by="t",
            args={"port": port, "path": "/chat"}))
        _, out = ctx.execute_tool(ToolRequest(
            project="http-error", tool="chat_say", action_class=ActionClass.ACTIVE,
            target=target_ref("127.0.0.1"), requested_by="t",
            args={"conversation_id": opened["conversation_id"], "port": port,
                  "message": "hello", "path": "/chat"}))

        assert out["error"].startswith("chat_say got HTTP 503")
        assert out["http_status"] == 503
        assert out["evidence_refs"], "the failed exchange must still be recorded"
        assert "reply" not in out, (
            "an error response must not offer a `reply` key for something to score")
    finally:
        srv.shutdown()
