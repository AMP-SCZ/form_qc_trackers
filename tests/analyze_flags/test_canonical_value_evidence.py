"""Canonical episode keys must ignore display-only observed values."""

from analyze_flags.canonicalize import (
    canonicalize_message,
    escape_specific_flag_value,
    explode_specific_flags,
    split_entry,
)


def test_proposed_evidence_does_not_fragment_the_canonical_episode_key():
    first = canonicalize_message(
        "field_a",
        "[RULE-001] Related fields disagree.\n"
        "Variables & values: field_a=7; controller=0",
    )
    second = canonicalize_message(
        "field_a",
        "[RULE-001] Related fields disagree.\n"
        "Variables & values: field_a=8; controller=1",
    )

    assert first == second
    assert "Variables & values" not in first


def test_sentence_final_observed_number_does_not_fragment_episode_key():
    first = canonicalize_message(
        "field_a", "field_a is equal to 7.")
    second = canonicalize_message(
        "field_a", "field_a is equal to 8.")

    assert first == second
    assert "<num>" in first


def test_invalid_guid_variants_do_not_fragment_episode_key():
    first = canonicalize_message(
        "chrguid_guid",
        "GUID in incorrect format. GUID was reported to be ndar-bad1.",
    )
    second = canonicalize_message(
        "chrguid_guid",
        "GUID in incorrect format. GUID was reported to be bad|foo:bar.",
    )

    assert first == second
    assert first.endswith("GUID was reported to be <ids>.")


def test_duplicate_fluid_conflict_list_does_not_fragment_episode_key():
    first = canonicalize_message(
        "chrblood_wbid1",
        (
            "Duplicate blood ID/barcode value (chrblood_wbid1 = abc-1) "
            "also found on other subject(s): p001 "
            "(chrblood_wbid1 at baseline)."
        ),
    )
    second = canonicalize_message(
        "chrblood_wbid1",
        (
            "Duplicate blood ID/barcode value (chrblood_wbid1 = xyz)-2) "
            "also found on other subject(s): ndar_bad-2 "
            "(chrblood_wbid2 at month1), another_subject "
            "(chrblood_wbid3 at month2)."
        ),
    )

    assert first == second
    assert "<ids>" in first
    assert "p001" not in first
    assert "another_subject" not in first


def test_qcv1_transport_round_trips_pipe_backslash_and_entry_like_text():
    raw_value = "bad|foo:bar\\tail%7C"
    encoded = escape_specific_flag_value(raw_value)
    rendered = (
        "chrguid_guid : GUID in incorrect format. GUID was reported to be "
        f"{encoded}. | field_b : Variable is blank."
    )

    assert encoded.startswith("@qcv1:")
    assert "|" not in encoded
    assert "\\" not in encoded
    entries = list(explode_specific_flags(rendered))
    assert len(entries) == 2
    assert entries[0][0] == "chrguid_guid"
    assert entries[0][1].endswith(f"{raw_value}.")
    assert entries[1][:2] == ("field_b", "Variable is blank.")


def test_legacy_raw_pipe_is_not_an_entry_boundary_without_variable_header():
    rendered = (
        "chrguid_guid : GUID in incorrect format. GUID was reported to be "
        "bad|foo:bar. | field_b : Variable is blank."
    )

    entries = list(explode_specific_flags(rendered))
    assert [entry[0] for entry in entries] == ["chrguid_guid", "field_b"]
    assert entries[0][1].endswith("bad|foo:bar.")


def test_historical_percent_escape_is_not_decoded_without_qcv1_tag():
    variable, message = split_entry(
        "chrguid_guid : GUID was literally %7C and %25."
    )

    assert variable == "chrguid_guid"
    assert message == "GUID was literally %7C and %25."
