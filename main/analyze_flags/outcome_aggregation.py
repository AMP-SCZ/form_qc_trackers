"""Collision-safe aggregation helpers for outcome calculations."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from outcome_inputs import OutcomeInputError


_BOILERPLATE_OUTCOME_COLUMNS = {"variable", "value", "data_type", "ID"}

_NUMERIC_SENTINEL_REPLACEMENTS = {
    r"^\s*1903-03-03(?:[T\s].*)?\s*$": "-300",
    r"^\s*1909-09-09(?:[T\s].*)?\s*$": "-900",
}


def _require_identical_event_index(
    reference,
    candidate: pd.Index,
    *,
    frame_index: int,
    frame_description: str,
) -> pd.Index:
    """Validate identical unique membership and retain the first frame's order."""
    if reference is None:
        return candidate
    reference_only_count = int((~reference.isin(candidate)).sum())
    candidate_only_count = int((~candidate.isin(reference)).sum())
    if (
        len(candidate) != len(reference)
        or reference_only_count
        or candidate_only_count
    ):
        raise OutcomeInputError(
            f"{frame_description} frames must have identical REDCap event "
            f"indexes; frame 0 count={len(reference)}, "
            f"frame {frame_index} count={len(candidate)}, "
            f"frame 0 only count={reference_only_count}, "
            f"frame {frame_index} only count={candidate_only_count}"
        )
    return reference


def coerce_outcome_numeric(
    values: pd.DataFrame, fill_type: str
) -> pd.DataFrame:
    """Coerce outcome inputs while preserving AMP-SCZ missing sentinels."""
    if fill_type not in {"int", "float"}:
        raise OutcomeInputError(
            f"Numeric outcome fill type must be int or float, got {fill_type!r}"
        )
    normalized = values.replace(_NUMERIC_SENTINEL_REPLACEMENTS, regex=True)
    numeric = normalized.apply(pd.to_numeric, errors="coerce")
    # Non-finite parses (e.g. the literal string 'inf') previously crashed
    # the int cast with a raw IntCastingNaNError; treat them as missing
    # (review 2026-08-22, C7).
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(-900)
    return numeric.astype(fill_type)


def reverse_outcome_numeric(
    values: pd.DataFrame, ceiling: float
) -> pd.DataFrame:
    """Reverse only in-range numeric items, never sentinels or codes."""
    numeric = coerce_outcome_numeric(values, "float")
    # Reversing anything outside [0, ceiling] fabricates plausible-looking
    # scores from missing codes ('-9' became ceiling+9; review 2026-08-22,
    # C7). Out-of-range values pass through unchanged so the downstream
    # sentinel checks on the raw frame still catch them.
    in_range = numeric.ge(0) & numeric.le(ceiling)
    reversed_values = ceiling - numeric
    return reversed_values.where(in_range, numeric)


def merge_named_outcome_columns(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Combine named result columns with an identical event-index contract."""
    if not frames:
        raise OutcomeInputError("At least one outcome frame is required")

    aligned = []
    used_columns: set[str] = set()
    reference_event_index = None
    for frame_index, frame in enumerate(frames):
        if "redcap_event_name" not in frame.columns:
            raise OutcomeInputError(
                "Outcome frame is missing required redcap_event_name"
            )
        missing_event_count = int(frame["redcap_event_name"].isna().sum())
        if missing_event_count:
            raise OutcomeInputError(
                f"Outcome frame contains missing REDCap events; frame "
                f"{frame_index} missing count={missing_event_count}"
            )
        if frame["redcap_event_name"].duplicated().any():
            raise OutcomeInputError("Outcome frame contains duplicate REDCap events")
        event_index = pd.Index(frame["redcap_event_name"])
        reference_event_index = _require_identical_event_index(
            reference_event_index,
            event_index,
            frame_index=frame_index,
            frame_description="Outcome",
        )
        payload = [
            column
            for column in frame.columns
            if column != "redcap_event_name"
            and column not in _BOILERPLATE_OUTCOME_COLUMNS
        ]
        overlap = used_columns.intersection(payload)
        if overlap:
            raise OutcomeInputError(
                f"Outcome frames contain duplicate named columns: {sorted(overlap)}"
            )
        used_columns.update(payload)
        aligned.append(
            frame.set_index("redcap_event_name")[payload].reindex(
                reference_event_index
            )
        )

    return pd.concat(aligned, axis=1, join="inner", sort=False).reset_index()


def combine_scid_conditions(
    frames: Sequence[pd.DataFrame], *, maximum: bool = False
) -> pd.DataFrame:
    """Combine SCID values under an identical event-index contract."""
    if not frames:
        raise OutcomeInputError("At least one SCID condition frame is required")

    values = []
    reference_event_index = None
    for index, frame in enumerate(frames):
        required = {"redcap_event_name", "value"}
        if not required.issubset(frame.columns):
            missing = sorted(required.difference(frame.columns))
            raise OutcomeInputError(
                f"SCID condition frame is missing required columns: {missing}"
            )
        missing_event_count = int(frame["redcap_event_name"].isna().sum())
        if missing_event_count:
            raise OutcomeInputError(
                f"SCID condition frame contains missing REDCap events; frame "
                f"{index} missing count={missing_event_count}"
            )
        if frame["redcap_event_name"].duplicated().any():
            raise OutcomeInputError(
                "SCID condition frame contains duplicate REDCap events"
            )
        event_index = pd.Index(frame["redcap_event_name"])
        reference_event_index = _require_identical_event_index(
            reference_event_index,
            event_index,
            frame_index=index,
            frame_description="SCID condition",
        )
        values.append(
            pd.to_numeric(
                frame.set_index("redcap_event_name")["value"],
                errors="coerce",
            ).rename(f"value_{index}").reindex(reference_event_index)
        )

    aligned = pd.concat(values, axis=1, join="inner", sort=False)
    if maximum:
        result = np.where(
            aligned.gt(0).any(axis=1),
            aligned.max(axis=1),
            np.where(aligned.eq(0).all(axis=1), 0, -900),
        )
    else:
        result = np.where(
            aligned.eq(1).any(axis=1),
            1,
            np.where(aligned.eq(0).all(axis=1), 0, -900),
        )
    return pd.DataFrame(
        {
            "redcap_event_name": aligned.index.to_list(),
            "value": result,
        }
    )
