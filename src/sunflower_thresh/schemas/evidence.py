"""Evidence items and the per-run manifest.

Evidence is content-addressed (stored by hash) so a finding is reproducible and
tamper-evident months later. The manifest records everything needed to rerun.
"""
from __future__ import annotations

import enum
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class EvidenceKind(str, enum.Enum):
    COMMAND = "command"
    HTTP_EXCHANGE = "http_exchange"
    PCAP = "pcap"
    SCREENSHOT = "screenshot"
    FINGERPRINT = "fingerprint"
    FIRMWARE_HASH = "firmware_hash"
    BINARY_REF = "binary_ref"
    CRASH = "crash"
    REBOOT_EVENT = "reboot_event"
    REPRODUCTION = "reproduction"
    LOG = "log"
    CVE_RECORD = "cve_record"
    DECISION = "decision"
    #: An inbound request the target made to our collector -- the target's own
    #: traffic, which is the only real proof of data leaving it.
    CALLBACK = "callback"


class EvidenceItem(BaseModel):
    """One piece of evidence. `sha256` addresses the raw bytes in the store."""

    kind: EvidenceKind
    sha256: str
    produced_by: str = Field(..., description="tool or agent that produced it")
    target: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    meta: dict = Field(default_factory=dict)
    # Provenance for external facts (NVD etc). Never allow an agent to invent these.
    source: str | None = None
    retrieved_at: str | None = None
    record_id: str | None = None


class RunManifest(BaseModel):
    """Makes a run reproducible: inputs, versions, models, hashes."""

    run_id: str
    project: str
    target: str = ""
    started_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    ended_at: str | None = None
    inventory_hash: str = ""
    tool_versions: dict[str, str] = Field(default_factory=dict)
    # role -> model id, so a run records exactly which line-up produced it.
    lineup: dict[str, str] = Field(default_factory=dict)
    # ...and role -> the name that actually SERVED the tokens, because the id
    # above is a catalog alias and aliases are reused. `local` is whatever
    # $HARNESS_LOCAL_MODEL pointed at on that machine that night, and
    # `claude-opus` named claude-opus-4-8 before it named claude-opus-5-5 --
    # so a ledger keyed only by the alias compares two different models under
    # one heading and calls it a trend.
    lineup_models: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    #: Which redaction policy was in force. Redaction happens before hashing, so
    #: the same exchange yields a different ref under a different policy -- without
    #: this, two runs compared across a policy change look like they found
    #: different evidence for the same thing. 0 means no redactor was configured.
    redaction_version: int = 0
    git_commit: str = ""
    #: The seed-vault revision the cases came from, and the case ids actually
    #: attempted. A transcript that cannot be traced to a corpus revision is weak
    #: evidence: the client asks "what exactly did you send", and "case INJ-D-0001 at
    #: corpus 4f9c1ab" is an answer where "case INJ-D-0001" is not -- the corpus is
    #: edited between engagements, which is the point of it.
    corpus_sha: str = ""
    corpus_cases: list[str] = Field(default_factory=list)
    #: {case id: why it was not attempted}. Persisted rather than printed, because a
    #: report that omits "we could not test indirect injection -- no sink was
    #: reachable" is misleading by omission, and the only defence against that is
    #: having the list in the record the report is generated from.
    skipped: dict[str, str] = Field(default_factory=dict)
    input_hashes: list[str] = Field(default_factory=list)
    output_hashes: list[str] = Field(default_factory=list)
    # Engagement + outcome, so the manifest is the whole record of a run.
    engagement: str = ""
    roe_hash: str = ""
    #: sha256 of the engagement's test plan as it was on disk when the run started.
    #: Sibling of `roe_hash` and `inventory_hash`, and the thing that makes plan drift
    #: detectable: the plan the client signed is rendered into the test-plan document
    #: from the same file, so a report built from a run whose plan hash does not match
    #: the plan in the engagement directory is a report about a different test than the
    #: one that was agreed. Empty when no plan was used.
    plan_hash: str = ""
    model: str = ""            # catalog alias; what `tally` keys history by
    model_name: str = ""       # what actually served the tokens
    objective: str = ""
    stop_reason: str = ""
    finding_ids: list[str] = Field(default_factory=list)
    validated: int = 0
    evidence_count: int = 0
    economics: dict = Field(default_factory=dict)
    # In-scope actions a disabled capability blocked -- surfaced for human review.
    missed_opportunities: list[dict] = Field(default_factory=list)

    def close(self) -> RunManifest:
        self.ended_at = datetime.now(UTC).isoformat()
        return self
