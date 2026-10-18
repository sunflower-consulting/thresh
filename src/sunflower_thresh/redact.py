"""Strip personal data BEFORE it is hashed, because the hash is the identity.

STATUS.md already names this: "evidence is immutable with no redaction, which
turns the tamper-evident store into a GDPR liability the first time a captured
body holds a real address. Redaction has to happen at capture, because an
artifact's identity is its hash."

Chatbot testing makes that acute rather than theoretical. A support bot's replies
are made of customer data, and the whole point of a disclosure test is to get it to
say some. Safety testing is worse: a harmful-content case that works produces
exactly the text nobody wants permanently content-addressed in a directory on a
laptop.

So this runs inside ``EvidenceStore.put_item``, before the bytes are hashed and
written. Two consequences worth stating plainly:

  * the stored artifact and the live reply have DIFFERENT hashes. That is the
    trade: a tamper-evident store of redacted bytes, rather than a tamper-evident
    store of personal data. `RunManifest.redaction_version` records which policy
    was in force so two runs remain comparable.
  * ``keep`` is load-bearing. Redacting a canary would destroy the oracle that
    proves the finding -- the seeded token is the evidence. Anything a case seeded
    is preserved verbatim, and that is a correctness requirement, not a nicety.

Deliberately four patterns, not a classifier. A regex that is wrong about a Luhn
check is a bug you can see; a model that is wrong about what counts as personal
data is a bug you cannot. Richer classification, per-jurisdiction rules, retention
sweeps and a purge command are all absent on purpose rather than done badly.
"""
from __future__ import annotations

import re

#: (name, pattern, replacement). Order matters only in that a later pattern sees
#: the earlier one's output.
PATTERNS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), "[REDACTED:email]"),
    # Bearer tokens, API keys and private-key blocks. Not personal data, but a
    # captured Authorization header is a live credential sitting in an immutable
    # store, which is its own liability.
    ("secret", re.compile(
        r"(?i)\b(?:bearer\s+[A-Za-z0-9._~+/-]{12,}=*"
        # The separator allows an optional closing quote and `=>` because the
        # commonest shape a captured body has is JSON: `{"api_key":"..."}` puts a
        # quote between the keyword and the colon, and the first version of this
        # pattern required them adjacent, so every JSON-bodied credential this
        # tool captured was stored verbatim. Measured against prose
        # ("my password is a secret", "tokenizer=fast") and short values
        # (`{"token": "abc"}`), none of which match.
        r"|(?:api[_-]?key|secret|password|passwd|token)['\"]?\s*(?:=>|[:=])\s*"
        r"['\"]?[^\s'\"&,}]{8,})"),
     "[REDACTED:secret]"),
    ("private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL), "[REDACTED:private_key]"),
    # E.164-ish, with enough structure not to eat every long number.
    ("phone", re.compile(r"(?<![\w.])\+[1-9]\d{1,3}[\s.-]?\(?\d{2,4}\)?"
                         r"(?:[\s.-]?\d{2,4}){1,3}(?![\w.])"), "[REDACTED:phone]"),
)

#: A 13-19 digit run, optionally spaced or dashed. Candidate only: it is redacted
#: solely if it passes a Luhn check, so an order number or a long id survives.
_CARD = re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?![\d.])")

#: Bumped whenever PATTERNS or the Luhn rule changes. Recorded in the run manifest,
#: because redaction changes hashes and two runs compared across a policy change
#: would otherwise look like they found different evidence for the same thing.
#: 2 widened the secret separator to cover JSON and `=>`; a run recorded under
#: version 1 captured credentials in a JSON body verbatim, so its evidence may
#: hold live secrets and cannot be compared to a version 2 run as though the
#: same policy had been in force.
REDACTION_VERSION = 2


def luhn_ok(digits: str) -> bool:
    """Luhn check, so a 16-digit order number is not mistaken for a card."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def redact(data: bytes | str, *, keep: tuple[str, ...] = ()) -> tuple[str, dict]:
    """Return (redacted text, counts per pattern).

    ``keep`` is protected by substitution: each kept value is swapped for a
    placeholder before the patterns run and restored afterwards. A canary that
    happens to look like a secret -- which a canary very often does, since it is a
    high-entropy token -- must come through untouched, or the oracle that proves
    the finding is destroyed by the thing meant to make the finding safe to store.
    """
    text = data.decode(errors="replace") if isinstance(data, bytes) else str(data)
    counts: dict[str, int] = {}

    holds: dict[str, str] = {}
    for i, value in enumerate(v for v in keep if v):
        placeholder = f"\x00KEEP{i}\x00"
        if value in text:
            holds[placeholder] = value
            text = text.replace(value, placeholder)

    for name, pattern, replacement in PATTERNS:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = counts.get(name, 0) + n

    def _card(match: re.Match) -> str:
        if luhn_ok(match.group(0)):
            counts["card"] = counts.get("card", 0) + 1
            return "[REDACTED:card]"
        return match.group(0)

    text = _CARD.sub(_card, text)

    for placeholder, value in holds.items():
        text = text.replace(placeholder, value)
    return text, counts


def make_redactor(keep: tuple[str, ...] = ()):
    """A redactor in the shape ``EvidenceStore`` wants: bytes -> (bytes, meta).

    Built as a closure rather than a subclass so a run can add its seeded canaries
    to ``keep`` as cases mint them, without reconstructing the store.
    """
    kept = list(keep)

    def _redactor(data: bytes) -> tuple[bytes, dict]:
        text, counts = redact(data, keep=tuple(kept))
        meta = {"version": REDACTION_VERSION}
        if counts:
            meta["counts"] = counts
        return text.encode(), meta

    def _keep(*values: str) -> None:
        """Protect further values -- a canary minted after the store was built."""
        for value in values:
            if value and value not in kept:
                kept.append(value)

    _redactor.keep = _keep          # type: ignore[attr-defined]
    _redactor.kept = kept           # type: ignore[attr-defined]
    return _redactor
