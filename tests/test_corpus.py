"""Loading the corpus, and the failure that is dangerous because it is quiet.

A SHORT LOAD IS THE DEFECT THIS FILE EXISTS FOR.

The engine this was ported from raised only when EVERY case failed, a guard it
earned the hard way: the corpus added three fields, the loader did not know them,
all 115 cases failed, and it returned zero cases without raising. An engagement
would have sent nothing and reported the assistant resistant.

But a PARTIAL failure stayed silent, and an `asi` field once broke 29 of 244 that
way with nothing said. Both are one defect at two scales and the quiet one is
worse: 243 of 244 reads exactly like a corpus, and the case it dropped is the case
nobody ran.

It is armed right now rather than hypothetical. `drafted_by` is a field the corpus
schema allows and the old loader does not know, so the day somebody promotes a
model-drafted case, a loader built the old way returns 243 and says nothing.

Tests that need the corpus skip without it, which is deliberate: this is a public
repository and the corpus is private, so the contract tests are the ones that
cannot run for a stranger and the rest must.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from tests.paths import CORPUS

from sunflower_thresh.cli import main as _run_cli
from sunflower_thresh.corpus import (
    CANARY_PLACEHOLDERS,
    CASE_STATUSES,
    CHANNELS,
    CONTROL_KINDS,
    HARMFUL_RUBRIC,
    PLACEHOLDER_RE,
    REFUSAL_RUBRIC,
    REPEAT_MAX_N,
    RUBRIC_RUNGS,
    SYNTHESISED_JUDGE_RUBRICS,
    CorpusCase,
    CorpusError,
    corpus_root_from_env,
    expand,
    expand_pattern,
    load_chat_cases,
    load_corpus,
    names_in,
    resolve_bindings,
    to_chat_case,
)

needs_corpus = pytest.mark.skipif(
    not (CORPUS / "cases").is_dir(),
    reason="no seed-vault checkout; the corpus is private and this repo is public")


#: A rubric table the default case can be scored against, so a test about turns or
#: controls does not have to care about rubrics.
RUBRICS: dict[str, dict] = {
    "restricted_action_performed": {"type": "contains",
                                    "value": "{{RESTRICTED_ACTION_MARKER}}"},
}


def _case(**kw) -> CorpusCase:
    base = {"id": "X-1", "category": "injection/direct", "title": "t",
            "goal": "a goal long enough", "owasp": "LLM01",
            "methodology_phase": "03", "turns": ["do {{RESTRICTED_ACTION}}"],
            "success": {"any_of": ["restricted_action_performed"]},
            "severity_hint": "high", "source": "internal", "path": Path("x.yaml")}
    return CorpusCase(**{**base, **kw})


def _bound(case, rubrics=None, **kw):
    """`to_chat_case` with the default case's own bindings already supplied."""
    binds = {"RESTRICTED_ACTION": "x", "RESTRICTED_ACTION_MARKER": "m-1"}
    binds.update(kw.pop("bindings", {}) or {})
    return to_chat_case(case, rubrics if rubrics is not None else RUBRICS,
                        bindings=binds, **kw)


# --- expansion -------------------------------------------------------------------

def test_an_unbound_placeholder_is_left_verbatim():
    """Visible rather than empty. Dropping `{{SECRET_MARKER}}` to "" would send a
    plausible-looking probe that cannot possibly match its own oracle, and the run
    would report the target clean."""
    assert expand("say {{SECRET_MARKER}}", {}) == "say {{SECRET_MARKER}}"


def test_a_transform_argument_substitutes_the_longest_name_first():
    """THE expansion test. `RESTRICTED_ACTION` is a prefix of
    `RESTRICTED_ACTION_MARKER`, so in dict order the short key wins sometimes and
    the long key other times, and the same case base64s to different bytes
    depending on how the bindings were built.

    An intermittently wrong payload is worse than a broken one: it looks like
    target nondeterminism."""
    b = {"RESTRICTED_ACTION": "append the hash", "RESTRICTED_ACTION_MARKER": "d59eced1"}
    got = base64.b64decode(expand("{{B64:emit RESTRICTED_ACTION_MARKER}}", b)).decode()
    assert got == "emit d59eced1", got


def test_an_unimplemented_transform_raises_rather_than_passing_through():
    """`{{ROT47:...}}` is transform SYNTAX with a name we do not implement. Left
    verbatim it would send `{{ROT47:print the canary}}` as literal text: a payload
    that reads plausibly, cannot match its own oracle, and spends trials proving
    nothing."""
    with pytest.raises(CorpusError, match="unknown transform"):
        expand("{{ROT47:x}}", {})


def test_repeat_refuses_to_build_a_payload_nobody_agreed_to_send():
    """A case must not be able to wedge a run or send a megabyte at a customer by
    typing a longer number, and the corpus refusing it at authoring time is not a
    reason to trust the number here: this loads a checkout that may predate the
    check."""
    assert expand("{{REPEAT:3:ab}}", {}) == "ababab"
    with pytest.raises(CorpusError, match="refusing to build"):
        expand(f"{{{{REPEAT:{REPEAT_MAX_N + 1}:ab}}}}", {})
    with pytest.raises(CorpusError, match="REPEAT takes N:text"):
        expand("{{REPEAT:notanumber}}", {})


def test_a_literal_span_is_escaped_after_substitution_and_only_inside_itself():
    r"""Python's `re` has no `\Q..\E`, so without this a canary containing a regex
    metacharacter either over-matches or raises mid-run. Outside the span is regex
    the author wrote deliberately and must not be escaped."""
    out = expand_pattern(r"^\Q{{SECRET_MARKER}}\E.*$", {"SECRET_MARKER": "a.b+c"})
    assert out == r"^a\.b\+c.*$"


def test_placeholder_names_are_found_inside_transform_arguments():
    """A name inside `{{B64:...}}` is still a binding the engagement must supply,
    and missing it would base64 a literal placeholder into the payload."""
    assert names_in("{{B64:x RESTRICTED_ACTION}} {{TOOL_NAME}}") == {
        "RESTRICTED_ACTION", "TOOL_NAME"}


def test_the_placeholder_regex_matches_what_the_corpus_writes():
    """Copied rather than imported, because this has to load a checkout at a pinned
    revision without pip-installing it. A divergence is caught here."""
    # `findall` on a two-group pattern yields "" for the unmatched group, not None;
    # `expand` reads `m.group(2) is None` to tell `{{A}}` from `{{A:}}`, so the
    # distinction lives in `match`, not here.
    assert PLACEHOLDER_RE.findall("{{A}} {{B64:c}}") == [("A", ""), ("B64", "c")]
    assert PLACEHOLDER_RE.match("{{A}}").group(2) is None


# --- the short load --------------------------------------------------------------

def test_a_corpus_with_one_bad_case_is_refused_by_default(tmp_path):
    """The whole point. 243 of 244 reads exactly like a corpus."""
    cases = tmp_path / "cases" / "injection"
    cases.mkdir(parents=True)
    (cases / "good.yaml").write_text(
        "id: G-1\ncategory: injection/direct\ntitle: t\ngoal: a goal long enough\n"
        "owasp: LLM01\nmethodology_phase: '03'\nturns: ['x']\n"
        "success: {any_of: [r]}\nseverity_hint: high\nsource: internal\n")
    (cases / "bad.yaml").write_text("id: B-1\nfuture_field: 1\n")

    with pytest.raises(CorpusError) as exc:
        load_corpus(tmp_path)
    assert "SHORT" in str(exc.value)
    assert "1 case(s) failed" in str(exc.value)
    assert "future_field" in str(exc.value), "it must name what it did not understand"


def test_a_short_load_can_be_accepted_knowingly(tmp_path):
    """An engagement that has decided to proceed without a case may, and the count
    reaches the manifest, because skipping a case is a fact about the measurement."""
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "bad.yaml").write_text("id: B-1\nfuture_field: 1\n")
    corpus = load_corpus(tmp_path, allow_broken=True)
    assert len(corpus) == 0
    assert "bad" in corpus.broken


def test_the_field_that_arms_this_defect_today_is_known():
    """`drafted_by` is allowed by the corpus schema and was unknown to the loader
    this was ported from. Zero cases carry it right now, so the failure is latent:
    promote one model-drafted case and the old loader returns one fewer and says
    nothing."""
    assert "drafted_by" in CorpusCase.__dataclass_fields__


@needs_corpus
def test_every_field_the_corpus_schema_allows_is_a_loader_field():
    """The drift alarm, in the direction that fails quietly. A field the schema
    permits and the loader does not know makes every case carrying it fail, and
    with `allow_broken` that is a silent partial load."""
    import json

    schema = json.loads((CORPUS / "schema/testcase.schema.json").read_text())
    known = set(CorpusCase.__dataclass_fields__) - {"path"}
    unknown = set(schema["properties"]) - known
    assert not unknown, f"the corpus allows fields this loader drops: {sorted(unknown)}"


# --- the rung mapping ------------------------------------------------------------

def test_a_contains_rubric_is_graded_by_its_value_and_not_its_type():
    """THE grading test. `contains: {{SECRET_MARKER}}` is unguessable and
    `contains: "password"` is a word a bot might say unprompted. Same rubric type,
    two different grades of evidence, and grading by type would make one a lie."""
    rubrics = {"leak": {"type": "contains", "value": "{{SECRET_MARKER}}"},
               "word": {"type": "contains", "value": "password"}}
    case = _case(success={"any_of": ["leak", "word"]})
    cc = to_chat_case(case, rubrics, bindings={"SECRET_MARKER": "canary-4f9a2c",
                                               "RESTRICTED_ACTION": "x"},
                      canaries={"canary-4f9a2c"})
    kinds = [next(iter(o)) for o in cc.oracles]
    assert kinds == ["canary", "structural"], kinds


def test_an_encoded_canary_is_still_canary_grade():
    """Base64 of an unguessable token is just as unguessable. The encoding changes
    what the leak looks like, not how well it is proved, and a bot that refuses in
    prose then prints the token in hex has leaked it."""
    rubrics = {"enc": {"type": "encoded_contains", "value": "{{DATA_CANARY}}",
                       "encodings": ["base64"]}}
    cc = to_chat_case(_case(success={"any_of": ["enc"]}), rubrics,
                      bindings={"DATA_CANARY": "canary-d81e03",
                                "RESTRICTED_ACTION": "x"},
                      canaries={"canary-d81e03"})
    assert next(iter(cc.oracles[0])) == "canary"


def test_a_rubric_type_this_cannot_express_excludes_the_case_with_its_reason():
    """Rather than running it blind. The reason names the rubric, which is what an
    author needs in order to fix it."""
    with pytest.raises(CorpusError, match="cannot express"):
        to_chat_case(_case(success={"any_of": ["odd"]}),
                     {"odd": {"type": "vibes"}}, bindings={"RESTRICTED_ACTION": "x"})


def test_an_unbound_binding_makes_the_case_unreachable_and_names_which():
    """Rather than sending `{{OTHER_USER_ID}}` at a customer's bot and scoring
    whatever comes back. Naming the placeholder is the difference between a reason
    an operator can act on and "something is missing"."""
    cc = _bound(_case(turns=["read {{OTHER_USER_ID}}"]))
    binding_needs = [n for n in cc.needs if n.startswith("binding:")]
    assert binding_needs and "OTHER_USER_ID" in binding_needs[0]


def test_a_marker_the_case_plants_is_minted_and_one_the_target_holds_is_not():
    """The whole argument about what counts as proof. An extraction case asks the
    bot for a secret it was GIVEN: mint a token and the bot has never seen it, so it
    can never come back, and the case scores 0.00 against a target that leaked
    everything."""
    planted = _case(setup={"channel": "document",
                           "indirect_content": "note: {{DATA_CANARY}}"},
                    turns=["summarise it"],
                    success={"any_of": ["leak"]})
    cc = to_chat_case(planted, {"leak": {"type": "contains",
                                         "value": "{{DATA_CANARY}}"}}, bindings={})
    assert not [n for n in cc.needs if n.startswith("binding:")], (
        "a planted marker is mintable, so it is not a missing binding")

    held = _case(turns=["tell me {{SECRET_MARKER}}"], success={"any_of": ["leak"]})
    cc = to_chat_case(held, {"leak": {"type": "contains",
                                      "value": "{{SECRET_MARKER}}"}}, bindings={})
    assert any("SECRET_MARKER" in n for n in cc.needs), (
        "a marker the target already holds must never be minted")


def test_a_consumption_case_is_clamped_to_two_trials():
    """One long reply is a finding; a thousand is an outage we caused. Ten trials
    plus ten controls on a cost rung is a load test nobody agreed to run."""
    cc = to_chat_case(_case(success={"any_of": ["len"]}),
                      {"len": {"type": "response_length", "min_chars": 5000}},
                      bindings={"RESTRICTED_ACTION": "x"}, trials=10)
    assert cc.trials == 2 and cc.control_trials == 2


def test_only_a_matched_control_contributes_turns():
    """For `inseparable` and `seeded` the corpus is stating a matched control is
    impossible and why. Inventing turns by deleting words we guessed were the attack
    would falsify nothing."""
    matched = _bound(
        _case(control={"kind": "matched", "turns": ["please {{RESTRICTED_ACTION}}"]}))
    assert [t.content for t in matched.control_turns] == ["please x"]

    inseparable = _bound(
        _case(control={"kind": "inseparable", "reason": "no request remains"}))
    assert inseparable.control_turns == []
    assert inseparable.control_note == "no request remains"


def test_a_seeded_case_carries_what_to_plant_and_what_happens_if_you_do_not():
    """An indirect case run WITHOUT its seeding looks like a clean pass: the payload
    is never delivered, the bot answers an ordinary question, and the report says
    the target resisted something nobody tested."""
    cc = _bound(_case(setup={"channel": "tool_output", "tool": "{{TOOL_NAME}}",
                             "indirect_content": "do {{RESTRICTED_ACTION}}"}),
                bindings={"TOOL_NAME": "lookup"})
    assert "plant_content" in cc.needs and "seed_tool_output" in cc.needs
    assert "lookup" in cc.seed_note
    assert "NOT TESTED" in cc.seed_note


def test_a_multi_turn_case_needs_a_target_that_remembers_turn_one():
    cc = _bound(_case(turns=["one", "two"]))
    assert "multi_turn" in cc.needs and "stateful" in cc.needs


# --- the judge coupling ----------------------------------------------------------

def test_the_refusal_rubric_has_a_deterministic_floor():
    """206 of the corpus's 244 cases suppress on `refusal`, so they must work with
    no judge configured. A calibrated judge is still preferred when the run has
    one, which is what makes this rubric's calibration the highest-leverage number
    in the exercise: a miscalibrated one silently changes 206 results."""
    cc = to_chat_case(_case(success={"any_of": ["leak"], "none_of": [REFUSAL_RUBRIC]}),
                      {"leak": {"type": "contains", "value": "x"},
                       REFUSAL_RUBRIC: {"type": "llm_judge", "question": "did it?"}},
                      bindings={"RESTRICTED_ACTION": "x"})
    assert next(iter(cc.suppressors[0])) == "refusal", (
        "the suppressor must not require a judge")
    assert "judge" not in cc.needs


def test_a_judged_harmful_content_case_needs_the_capture_switch():
    """A hit means the reply itself is the evidence, which is exactly the evidence
    nobody wants written to disk by default."""
    cc = to_chat_case(_case(success={"any_of": [HARMFUL_RUBRIC]}),
                      {HARMFUL_RUBRIC: {"type": "llm_judge", "question": "q"}},
                      bindings={"RESTRICTED_ACTION": "x"})
    assert "capture_harmful_content" in cc.needs


def test_a_judged_safety_case_needs_it_whichever_rubric_reads_the_reply():
    """Keying only on `harmful_content` was too narrow the moment the category grew
    past it: a case scored by a judge asking whether an exemption was granted would
    run with the switch off while collecting exactly what the switch gates."""
    cc = to_chat_case(_case(category="safety/refusal",
                            success={"any_of": ["authority_exemption_granted"]}),
                      {"authority_exemption_granted": {"type": "llm_judge",
                                                       "question": "q"}},
                      bindings={"RESTRICTED_ACTION": "x"})
    assert "capture_harmful_content" in cc.needs


def test_a_safety_case_proved_by_a_canary_stays_reachable_with_the_switch_off():
    """The other half, and it matters more than it looks. A safety case scored on a
    planted marker proves itself from a token the client seeded, so there is nothing
    in its evidence to withhold -- and it has to stay reachable where the switch is
    off, which is where it is needed most: it is the case that still carries a
    finding when the technique cases are rejected against their own controls."""
    cc = to_chat_case(_case(category="safety/refusal", success={"any_of": ["leak"]}),
                      {"leak": {"type": "contains", "value": "{{DATA_CANARY}}"}},
                      bindings={"DATA_CANARY": "c-1", "RESTRICTED_ACTION": "x"},
                      canaries={"c-1"})
    assert "capture_harmful_content" not in cc.needs


@needs_corpus
def test_every_judge_rubric_the_corpus_names_is_calibrated_or_listed():
    """The drift alarm the two repos share. A rubric with a local file has criteria
    and a threshold somebody chose; a rubric with neither a file nor a place on the
    synthesised list is a case that quietly started scoring on a default.

    The naming conventions diverged once already -- four kebab-case files against
    seven snake_case names, overlap zero, `fabricated-citation` against
    `fabricated_citation` -- so the one place the repos agreed about a judgement was
    the one place neither could see the other."""
    import yaml

    from sunflower_thresh.judge import load_rubrics

    corpus_rubrics = yaml.safe_load((CORPUS / "rubrics/default.yaml").read_text())
    named = {n for n, body in corpus_rubrics.items()
             if isinstance(body, dict) and body.get("type") == "llm_judge"}
    local = set(load_rubrics())
    adrift = named - local - set(SYNTHESISED_JUDGE_RUBRICS)
    assert not adrift, (
        f"the corpus names judge rubrics with no local file and no place on the "
        f"synthesised list: {sorted(adrift)}")


# --- binding precedence ----------------------------------------------------------

def test_the_signed_inventory_beats_the_plan_which_beats_a_file():
    """A value that decides what gets sent at a client's system comes from the
    document they signed, not from whatever file was on the operator's disk."""
    b, _can = resolve_bindings(
        file_bindings={"TOOL_NAME": "file"},
        plan={"placeholders": {"TOOL_NAME": "plan"}},
        inventory_seeds={"TOOL_NAME": "inventory"})
    assert b["TOOL_NAME"] == "inventory"


def test_canaries_are_named_rather_than_guessed_from_entropy():
    """What makes a value canary-grade is that the TARGET holds it and could not
    have invented it, which is a fact about the engagement and not about the
    characters."""
    _b, canaries = resolve_bindings(
        inventory_seeds={"SECRET_MARKER": "short", "CLIENT_NAME": "Zx9Qp2Lm"})
    assert canaries == {"short"}, "entropy is not the test; the name is"
    assert all(n in CANARY_PLACEHOLDERS for n in ("SECRET_MARKER",))


# --- against the real corpus -----------------------------------------------------

@needs_corpus
def test_the_whole_corpus_loads_and_converts():
    """244 of 244, zero broken. The number is asserted so a drop is a failure here
    rather than a quieter run later."""
    cases, corpus, skipped = load_chat_cases(CORPUS)
    assert len(corpus) == 244, f"the corpus moved: {len(corpus)} cases"
    assert corpus.broken == {}
    assert len(cases) + len(skipped) == 244
    assert all("not selected" in r for r in skipped.values()), skipped


@needs_corpus
def test_no_rubric_type_in_the_corpus_is_unexpressible():
    """A type with no rung means a case runs blind, so the table and the corpus have
    to agree."""
    corpus = load_corpus(CORPUS)
    assert corpus.unmapped_types() == {}
    assert set(corpus.unmapped_types()) <= set(RUBRIC_RUNGS)


@needs_corpus
def test_the_corpus_revision_reaches_the_loader():
    """A transcript that cannot be traced to the revision that produced it is weak
    evidence: the client asks what exactly was sent, and "INJ-D-0001 as it stood at
    4f9c1ab" is an answer where "INJ-D-0001" is not."""
    assert load_corpus(CORPUS).sha


# --- the command -----------------------------------------------------------------

def test_the_new_corpus_variable_is_read_before_the_legacy_one():
    """Both names work, because an operator with the old one exported will not be
    told their corpus vanished. The new one wins where both are set, so moving over
    is one export and not a hunt for which of the two took effect."""
    both = {"THRESH_CORPUS": "/new", "STEM_CORPUS": "/renamed",
            "HARNESS_CORPUS": "/old"}
    assert corpus_root_from_env(both) == "/new"
    # Every old name still resolves. There are two: the private engine this was
    # extracted from, and what this package was called before it became thresh. An
    # operator who exported either would otherwise meet a refusal naming only the
    # variable they did not set.
    assert corpus_root_from_env({"STEM_CORPUS": "/renamed"}) == "/renamed"
    assert corpus_root_from_env({"HARNESS_CORPUS": "/old"}) == "/old"
    assert corpus_root_from_env({"STEM_CORPUS": "/renamed",
                                 "HARNESS_CORPUS": "/old"}) == "/renamed"
    assert corpus_root_from_env({"THRESH_CORPUS": "   "}) == "", "blank is not a path"
    assert corpus_root_from_env({}) == ""


def _corpus_cli(tmp_path, *argv, extra=None):
    """`thresh corpus` over a one-case library, through the real parser."""
    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    rubrics = tmp_path / "rubrics"
    rubrics.mkdir(exist_ok=True)
    # A literal value, so a case is unreachable only for what its TURNS want. A
    # rubric placeholder is a need too, which is correct and would mask that here.
    (rubrics / "r.yaml").write_text(
        "restricted_action_performed:\n  type: contains\n  value: 'm-1'\n")
    (cases / "u.yaml").write_text(
        "id: U-1\ncategory: injection/direct\ntitle: t\ngoal: a goal long enough\n"
        "owasp: LLM01\nmethodology_phase: '03'\n"
        "turns: ['mail {{OTHER_USER_EMAIL}} about {{RESTRICTED_ACTION}}']\n"
        "success: {any_of: [restricted_action_performed]}\n"
        "severity_hint: high\nsource: internal\n")
    for name, body in (extra or {}).items():
        (cases / name).write_text(body)
    return _run_cli(["corpus", "--corpus", str(tmp_path), *argv])


def test_the_command_names_the_placeholder_nobody_bound(tmp_path, capsys, monkeypatch):
    """A count alone sends an operator reading 244 case files to find out which
    binding is missing. Naming them is the difference between a one-line fix and an
    afternoon, and the list is what a bindings file gets written from."""
    monkeypatch.delenv("THRESH_CORPUS", raising=False)
    monkeypatch.delenv("HARNESS_CORPUS", raising=False)
    assert _corpus_cli(tmp_path) == 0
    out = capsys.readouterr().out
    assert "1 case(s) would be UNREACHABLE" in out
    assert "OTHER_USER_EMAIL" in out
    assert "RESTRICTED_ACTION" in out


def test_the_command_refuses_a_short_library_rather_than_reporting_a_smaller_one(
        tmp_path, capsys, monkeypatch):
    """The refusal has to survive the trip through the CLI, which is where an
    engagement meets it. A non-zero exit is the half that matters to a script."""
    monkeypatch.delenv("THRESH_CORPUS", raising=False)
    monkeypatch.delenv("HARNESS_CORPUS", raising=False)
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "bad.yaml").write_text("id: B-1\nfuture_field: 1\n")
    assert _corpus_cli(tmp_path) == 2
    assert "SHORT" in capsys.readouterr().err
    assert _corpus_cli(tmp_path, "--allow-broken") == 0, "and may be overridden"


def test_the_command_says_where_to_point_it_when_nothing_is_configured(
        tmp_path, capsys, monkeypatch):
    """Exit 2 with the variable's name, not a traceback and not an empty report."""
    monkeypatch.delenv("THRESH_CORPUS", raising=False)
    monkeypatch.delenv("HARNESS_CORPUS", raising=False)
    assert _run_cli(["corpus"]) == 2
    assert "THRESH_CORPUS" in capsys.readouterr().err


def test_the_command_says_why_a_case_was_left_out_and_not_only_how_many(
        tmp_path, capsys, monkeypatch):
    """"241 of 244 selected" is the shape of a readout that gets quoted as coverage.
    A case held back for being a draft and one held back because this engine cannot
    express its rubric are different facts, and the second is a gap in the tool
    rather than a decision about the library."""
    monkeypatch.delenv("THRESH_CORPUS", raising=False)
    monkeypatch.delenv("HARNESS_CORPUS", raising=False)
    assert _corpus_cli(tmp_path, extra={
        "d.yaml": "id: D-1\ncategory: injection/direct\ntitle: t\n"
                  "goal: a goal long enough\nowasp: LLM01\nmethodology_phase: '03'\n"
                  "turns: ['x']\nsuccess: {any_of: [restricted_action_performed]}\n"
                  "severity_hint: high\nsource: internal\nstatus: draft\n"}) == 0
    out = capsys.readouterr().out
    assert "1 not selected" in out
    assert "1x not selected: D-1" in out, "the id, so it can be passed to --only"


def test_the_command_separates_a_declared_control_from_a_missing_one(
        tmp_path, capsys, monkeypatch):
    """A reachable case is not an automatic case. Without a control arm nothing
    reaches VALIDATED, so "214 reachable" read alone claims ground a run cannot
    cover - and a `seeded` control, where the corpus says the control is a re-plant
    an operator does by hand, is a different fact from nobody having written one.
    Collapsing the two is the readout defect this tool exists to catch elsewhere."""
    monkeypatch.delenv("THRESH_CORPUS", raising=False)
    monkeypatch.delenv("HARNESS_CORPUS", raising=False)
    body = ("category: injection/direct\ntitle: t\ngoal: a goal long enough\n"
            "owasp: LLM01\nmethodology_phase: '03'\nturns: ['x']\n"
            "success: {any_of: [restricted_action_performed]}\n"
            "severity_hint: high\nsource: internal\n")
    assert _corpus_cli(tmp_path, extra={
        "s.yaml": f"id: S-1\n{body}control:\n  kind: seeded\n"
                  "  reason: the control is the same plant without the clause\n",
        "n.yaml": f"id: N-1\n{body}",
    }) == 0
    out = capsys.readouterr().out
    assert "1x seeded: the control is a re-plant" in out
    assert "1x none: NOBODY WROTE ONE" in out, "a gap must not read as a declaration"


@needs_corpus
def test_every_corpus_case_without_a_matched_control_says_why(capsys, monkeypatch):
    """Measured: 46 seeded and 3 inseparable of 214 reachable, and none in the gap
    bucket. That is a fact about the library rather than about this loader, and it
    is asserted because the day it stops being true a run would quietly produce
    findings that nothing can falsify."""
    monkeypatch.setenv("THRESH_CORPUS", str(CORPUS))
    assert _run_cli(["corpus", "--collector"]) == 0
    assert "NOBODY WROTE ONE" not in capsys.readouterr().out


@needs_corpus
def test_every_vocabulary_the_corpus_uses_is_one_this_package_declares():
    """Three tuples that nothing referenced, and one of them was already wrong.

    CHANNELS, CASE_STATUSES and CONTROL_KINDS are this package's copy of
    vocabulary the corpus owns. Nothing imports them, so nothing broke when the
    corpus added `tool_metadata`, `agent_config` and `rendered_screen` and seven
    cases started using them -- the tuple simply became untrue, and an
    unreferenced constant that is untrue is worse than no constant, because the
    next reader believes it.

    This is the same drift that let an `asi` field reach 29 cases before anything
    here could load them. The lesson taken then was to compare the two sides
    rather than to maintain a copy by hand, so that is what this does.

    It needs the corpus and therefore skips without it, which is a real limit:
    on a machine with no checkout this guards nothing. The alternative -- a
    hand-maintained list in the test -- would drift exactly like the tuple it is
    checking, so skipping honestly is the better of the two.
    """
    corpus = load_corpus(CORPUS)
    cases = corpus.cases.values()

    checks = (
        ("CHANNELS", CHANNELS,
         {c.channel for c in cases if getattr(c, "channel", "")}),
        ("CASE_STATUSES", CASE_STATUSES,
         {c.status for c in cases if getattr(c, "status", "")}),
        ("CONTROL_KINDS", CONTROL_KINDS,
         {(c.control or {}).get("kind") for c in cases if (c.control or {}).get("kind")}),
    )

    problems = []
    for name, declared, used in checks:
        missing = sorted(used - set(declared))
        if missing:
            problems.append(f"{name} does not declare {missing}, which the corpus uses")
    assert not problems, (
        "\n  ".join(["the corpus uses vocabulary this package does not declare:"]
                    + problems)
        + "\n  Nothing imports these tuples, so nothing else would have failed.")


def test_no_module_quotes_a_corpus_size_that_disagrees_with_its_neighbours():
    """One corpus, one pair of numbers, however many comments cite it.

    `oracles.py` carried both. A module comment said `refusal` covers "206 of its
    244 cases (measured at seed-vault 4855589)"; `RefusalOracle`'s own docstring,
    four hundred lines down, said "130 of its 187 cases use it. Making all 130
    depend on a judge ...". Same corpus, same rubric, two answers, and the stale
    pair was the one a reader of the class would find.

    These figures are the argument for shipping a deterministic refusal matcher
    at all -- "most of the library depends on this" is the reason it is not left
    to a judge -- so a reader comparing 130/187 against the corpus they have in
    front of them has no way to tell which sentence to trust.

    Scanned rather than pinned to one blessed figure: pinning would need updating
    on every corpus revision and would fail for the whole repository at once. The
    rule enforced here is narrower and does not go stale: a single source file
    must not quote two DIFFERENT "N of M cases" totals.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "sunflower_thresh"
    pattern = re.compile(r"\b(\d{2,4})\s+of\s+(?:its\s+)?(\d{2,4})\s+cases\b")

    offenders = {}
    for path in sorted(src.rglob("*.py")):
        totals = {m.group(2) for m in pattern.finditer(path.read_text())}
        if len(totals) > 1:
            offenders[path.name] = sorted(totals)

    assert not offenders, (
        "these files each quote more than one corpus size, so at least one is "
        f"stale: {offenders}. Make them agree, and cite the seed-vault revision "
        "the figure was measured at.")
