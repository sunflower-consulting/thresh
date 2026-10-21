"""Which denials the sweep retries, and why the list must stay short.

`sweep.py` said "tests/test_message_budget.py asserts the contents of this
tuple". There is no such file in this repository, so `TRANSIENT_DENIALS` -- the
tuple that decides whether a refused case is re-sent up to MAX_RATE_RETRIES
times -- was unpinned while a comment vouched for it. An unpinned retry list is
a budget risk in the direction that costs a customer messages: a denial that
cannot clear, listed here, is re-asked six times per case.
"""
from __future__ import annotations

from sunflower_thresh.policy.decision import DenyReason
from sunflower_thresh.sweep import MAX_RATE_RETRIES, TRANSIENT_DENIALS


def test_only_denials_that_clear_by_themselves_are_retried():
    """A retry is a bet that waiting changes the answer.

    `rate_limited` self-clears: wait, retry, it works. Nothing else in
    `DenyReason` does. A spent message budget, an inactive authorisation, a
    thrown kill switch, an out-of-scope target -- each is a decision for the rest
    of the run, and retrying it spends the window re-asking a question whose
    answer cannot change.
    """
    assert TRANSIENT_DENIALS == (DenyReason.RATE_LIMITED.value,), (
        f"TRANSIENT_DENIALS is {TRANSIENT_DENIALS}; every member is re-sent up to "
        f"{MAX_RATE_RETRIES} times per case, so a denial that cannot clear belongs "
        f"nowhere near this tuple")


def test_every_retried_name_is_a_real_deny_reason():
    """A misspelt entry here fails silently in the safe direction -- nothing is
    ever retried -- which is exactly why nothing would report it. The gate's own
    vocabulary is the check."""
    known = {r.value for r in DenyReason}
    unknown = sorted(set(TRANSIENT_DENIALS) - known)
    assert not unknown, (
        f"{unknown} are not DenyReason values, so the sweep compares deny codes "
        f"against strings nothing can ever equal: {sorted(known)}")


def test_the_denials_that_must_never_be_retried_are_not():
    """Named one by one rather than as "everything else", so adding a reason to
    `DenyReason` does not quietly add it to the retry set by omission."""
    never = (DenyReason.MESSAGE_BUDGET_EXHAUSTED, DenyReason.AUTH_INACTIVE,
             DenyReason.KILLSWITCH, DenyReason.INVENTORY_UNVERIFIED,
             DenyReason.NO_INVENTORY)
    for reason in never:
        assert reason.value not in TRANSIENT_DENIALS, (
            f"{reason.value} is retried, and it does not clear on its own: the run "
            f"would sleep and re-send {MAX_RATE_RETRIES} times per case for a "
            f"refusal that is settled")


def test_the_retry_budget_is_bounded_and_worth_the_messages():
    """MAX_RATE_RETRIES multiplies the target messages a case can cost. It has to
    be small enough that a forty-case library cannot turn into an unbounded
    re-send, and the figure is quoted in the comment above the tuple."""
    assert 1 <= MAX_RATE_RETRIES <= 10, (
        f"MAX_RATE_RETRIES is {MAX_RATE_RETRIES}; a case can cost that many extra "
        f"passes over its whole trial budget")
