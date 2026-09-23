"""Build outcome-calculation subject frames from combined REDCap CSVs."""

import csv
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd


OUTCOME_TIMEPOINTS = (
    "screening",
    "baseline",
    *(f"month{x}" for x in range(1, 13)),
    "month18",
    "month24",
    "floating",
    "conversion",
)


class OutcomeInputError(ValueError):
    """Raised when combined CSV input cannot safely feed the outcomes."""


class CombinedSubjectStore(Mapping[str, pd.DataFrame]):
    """Lazy subject views over dense per-timepoint combined CSV frames.

    Keeping the source slices separate avoids materializing a network-wide,
    sparse union DataFrame. Only the requested subject is concatenated into
    the outcome engine's former one-row-per-event shape.
    """

    def __init__(
        self, frames: Sequence[pd.DataFrame], subject_ids: Sequence[str]
    ) -> None:
        self._frames = tuple(frames)
        self._subject_ids = tuple(subject_ids)
        self._subject_id_set = frozenset(self._subject_ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self._subject_ids)

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __contains__(self, subject_id: object) -> bool:
        return str(subject_id).strip() in self._subject_id_set

    def __getitem__(self, subject_id: str) -> pd.DataFrame:
        normalized_id = str(subject_id).strip()
        if normalized_id not in self._subject_id_set:
            raise KeyError(normalized_id)

        subject_slices = [
            frame.loc[[normalized_id]].reset_index(drop=True)
            for frame in self._frames
            if normalized_id in frame.index
        ]
        return pd.concat(subject_slices, ignore_index=True, sort=False)


def normalize_network(network: str) -> str:
    normalized = str(network).strip().replace("-", "").replace("_", "").upper()
    if normalized == "PRONET":
        return "PRONET"
    if normalized == "PRESCIENT":
        return "PRESCIENT"
    raise OutcomeInputError(
        f"Unsupported network {network!r}; expected PRONET or PRESCIENT"
    )


def normalize_timepoint(timepoint: str) -> str:
    normalized = (
        str(timepoint).strip().lower().replace("floating_forms", "floating")
    )
    month_match = re.fullmatch(r"month_?(\d+)", normalized)
    if month_match:
        normalized = f"month{int(month_match.group(1))}"
    if normalized not in OUTCOME_TIMEPOINTS:
        raise OutcomeInputError(f"Unsupported outcome timepoint {timepoint!r}")
    return normalized


def timepoint_file_token(timepoint: str) -> str:
    normalized = normalize_timepoint(timepoint)
    if normalized == "floating":
        return "floating_forms"
    return re.sub(r"^month(\d+)$", r"month_\1", normalized)


def network_file_token(network: str) -> str:
    return "ProNET" if normalize_network(network) == "PRONET" else "PRESCIENT"


def combined_csv_file(
    combined_csv_path: str | Path, network: str, timepoint: str
) -> Path:
    """Return the canonical combined REDCap CSV for one input slice."""
    return Path(combined_csv_path) / (
        f"AMPSCZ-combined-redcap_{timepoint_file_token(timepoint)}_"
        f"{network_file_token(network)}-day1to1.csv"
    )


def _cohort_arm(value) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "1.0", "chr"}:
        return "1"
    if normalized in {"2", "2.0", "hc"}:
        return "2"
    return ""


def _event_arm(event_name) -> str:
    match = re.search(r"_arm_(\d+)$", str(event_name or ""), flags=re.I)
    return match.group(1) if match else ""


def _event_timepoint(event_name) -> str:
    without_arm = re.sub(
        r"_arm_\d+$", "", str(event_name or "").strip(), flags=re.I
    )
    try:
        return normalize_timepoint(without_arm)
    except OutcomeInputError:
        return ""


def _event_name_for_timepoint(timepoint: str, arm: str) -> str:
    if not arm:
        return ""
    return f"{timepoint_file_token(timepoint)}_arm_{arm}"


def _canonical_event(event_name: str, timepoint: str) -> str:
    arm = _event_arm(event_name)
    if not arm:
        return event_name
    return _event_name_for_timepoint(timepoint, arm)


def _validate_csv_structure(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise OutcomeInputError(
                    f"Combined CSV is empty: {path}"
                ) from None

            duplicates = sorted(
                name for name, count in Counter(header).items() if count > 1
            )
            if duplicates:
                raise OutcomeInputError(
                    "Combined CSV has duplicate header(s) "
                    f"{duplicates[:20]}: {path}"
                )

            expected_fields = len(header)
            for row in reader:
                if len(row) != expected_fields:
                    raise OutcomeInputError(
                        "Combined CSV row has "
                        f"{len(row)} field(s); expected {expected_fields} "
                        f"at physical line {reader.line_num}: {path}"
                    )
    except FileNotFoundError:
        raise OutcomeInputError(f"Missing combined CSV: {path}") from None
    except csv.Error as exc:
        raise OutcomeInputError(
            f"Could not parse combined CSV {path} at physical line "
            f"{reader.line_num}: {exc}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise OutcomeInputError(f"Could not parse combined CSV {path}: {exc}") from exc


def _read_combined_csv(path: Path) -> pd.DataFrame:
    _validate_csv_structure(path)
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            on_bad_lines="error",
            low_memory=False,
        )
    except FileNotFoundError:
        raise OutcomeInputError(f"Missing combined CSV: {path}") from None
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise OutcomeInputError(f"Could not parse combined CSV {path}: {exc}") from exc

    if frame.empty:
        raise OutcomeInputError(f"Combined CSV contains zero participant rows: {path}")
    if "subjectid" not in frame.columns:
        raise OutcomeInputError(
            f"Combined CSV is missing required 'subjectid' column: {path}"
        )

    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].str.strip()
    if frame["subjectid"].eq("").any():
        raise OutcomeInputError(f"Combined CSV contains a blank subjectid: {path}")
    return frame


def _validate_slice_metadata(
    frame: pd.DataFrame, network: str, timepoint: str, path: Path
) -> None:
    checks = (
        ("network", normalize_network, network),
        ("timepoint", normalize_timepoint, timepoint),
    )
    for column, normalizer, expected in checks:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        for value in values[values.ne("")].unique():
            try:
                actual = normalizer(value)
            except OutcomeInputError:
                raise OutcomeInputError(
                    f"Combined CSV has invalid row-level {column} metadata: "
                    f"{path}"
                ) from None
            if actual != expected:
                raise OutcomeInputError(
                    f"Combined CSV row-level {column} metadata conflicts with "
                    f"selected {expected!r}: {path}"
                )
def _validate_no_repeat_rows(frame: pd.DataFrame, path: Path) -> None:
    repeating = pd.Series(False, index=frame.index, dtype=bool)
    for column in ("redcap_repeat_instrument", "redcap_repeat_instance"):
        if column in frame.columns:
            values = frame[column].fillna("").astype(str).str.strip()
            repeating |= values.ne("")
    repeat_count = int(repeating.sum())
    if repeat_count:
        raise OutcomeInputError(
            "Outcome calculations do not support repeating REDCap rows; "
            f"found {repeat_count} row(s): {path}"
        )




def load_combined_subjects(
    combined_csv_path: str | Path,
    network: str,
    *,
    timepoints: Sequence[str] = OUTCOME_TIMEPOINTS,
    subject_ids: Optional[Sequence[str]] = None,
) -> CombinedSubjectStore:
    """Load one network's combined CSVs into lazy outcome-ready subject views.

    Combined exports are split by timepoint. This restores the former JSON
    shape of one row per REDCap event, preserves missing-code strings, and
    converts only ordinary blank cells to NaN.
    """
    normalized_network = normalize_network(network)
    normalized_timepoints = tuple(normalize_timepoint(tp) for tp in timepoints)
    if not normalized_timepoints:
        raise OutcomeInputError("At least one outcome timepoint is required")
    if len(set(normalized_timepoints)) != len(normalized_timepoints):
        raise OutcomeInputError("Outcome timepoints must be unique")

    requested_ids = None
    if subject_ids is not None:
        requested_ids = {str(subject_id).strip() for subject_id in subject_ids}
        if "" in requested_ids:
            raise OutcomeInputError("Requested subject IDs cannot be blank")

    slices: list[pd.DataFrame] = []
    subject_order: dict[str, None] = {}
    explicit_arms_by_subject: dict[str, set[str]] = {}
    cohort_arms_by_subject: dict[str, set[str]] = {}
    for timepoint in normalized_timepoints:
        csv_path = combined_csv_file(
            combined_csv_path, normalized_network, timepoint
        )
        frame = _read_combined_csv(csv_path)
        _validate_slice_metadata(frame, normalized_network, timepoint, csv_path)
        _validate_no_repeat_rows(frame, csv_path)
        if requested_ids is not None:
            frame = frame[frame["subjectid"].isin(requested_ids)].copy()

        canonical_event_columns: dict[str, pd.Series] = {}
        for event_column in ("redcap_event_name", "event_name"):
            if event_column not in frame.columns:
                continue
            candidate = frame[event_column].fillna("").astype(str).str.strip()
            conflict = candidate.ne("") & candidate.map(_event_timepoint).ne(
                timepoint
            )
            if conflict.any():
                raise OutcomeInputError(
                    f"Combined CSV {event_column} conflicts with timepoint "
                    f"{timepoint!r} for {int(conflict.sum())} row(s): "
                    f"{csv_path}"
                )
            canonical_event_columns[event_column] = candidate.map(
                lambda event: _canonical_event(event, timepoint)
            )

        event_columns = ("redcap_event_name", "event_name")
        if all(column in canonical_event_columns for column in event_columns):
            redcap_events = canonical_event_columns["redcap_event_name"]
            legacy_events = canonical_event_columns["event_name"]
            both_present = redcap_events.ne("") & legacy_events.ne("")
            disagreement = both_present & redcap_events.ne(legacy_events)
            if disagreement.any():
                raise OutcomeInputError(
                    "Combined CSV redcap_event_name and event_name disagree "
                    f"for {int(disagreement.sum())} row(s): {csv_path}"
                )

        explicit_event = pd.Series("", index=frame.index, dtype=object)
        for event_column in event_columns:
            candidate = canonical_event_columns.get(event_column)
            if candidate is not None:
                explicit_event = explicit_event.mask(
                    explicit_event.eq(""), candidate
                )
        if "chrcrit_part" in frame.columns:
            explicit_arms = explicit_event.map(_event_arm)
            cohort_arms = frame["chrcrit_part"].map(_cohort_arm)
            row_arm_conflict = (
                explicit_arms.isin({"1", "2"})
                & cohort_arms.ne("")
                & explicit_arms.ne(cohort_arms)
            )
            if row_arm_conflict.any():
                raise OutcomeInputError(
                    "Combined CSV explicit event arm conflicts with "
                    f"chrcrit_part for {int(row_arm_conflict.sum())} row(s): "
                    f"{csv_path}"
                )
        frame["redcap_event_name"] = explicit_event
        frame["_outcome_timepoint"] = timepoint
        for subject_id in frame["subjectid"]:
            subject_order.setdefault(str(subject_id), None)
        for subject_id, event in zip(
            frame["subjectid"], frame["redcap_event_name"]
        ):
            arm = _event_arm(event)
            if arm:
                explicit_arms_by_subject.setdefault(str(subject_id), set()).add(arm)
        if "chrcrit_part" in frame.columns:
            for subject_id, cohort in zip(
                frame["subjectid"], frame["chrcrit_part"]
            ):
                arm = _cohort_arm(cohort)
                if arm:
                    cohort_arms_by_subject.setdefault(
                        str(subject_id), set()
                    ).add(arm)
        slices.append(frame)

    mixed_arm_subjects = sum(
        len(arms) > 1 for arms in explicit_arms_by_subject.values()
    )
    if mixed_arm_subjects:
        raise OutcomeInputError(
            "Outcome calculations do not support mixed REDCap arm histories; "
            f"found {mixed_arm_subjects} subject(s) with explicit arm transitions"
        )

    resolved_arms: dict[str, str] = {}
    for subject_id in subject_order:
        explicit_arms = explicit_arms_by_subject.get(subject_id, set())
        cohort_arms = cohort_arms_by_subject.get(subject_id, set())
        if len(explicit_arms) == 1:
            resolved_arms[subject_id] = next(iter(explicit_arms))
        elif not explicit_arms and len(cohort_arms) == 1:
            resolved_arms[subject_id] = next(iter(cohort_arms))

    prepared_slices: list[pd.DataFrame] = []
    unresolved_subjects: set[str] = set()
    duplicate_count = 0
    final_arms_by_subject: dict[str, set[str]] = {}
    for frame in slices:
        event_arms = frame["redcap_event_name"].map(_event_arm)
        invalid_arm = event_arms.ne("") & ~event_arms.isin({"1", "2"})
        if invalid_arm.any():
            raise OutcomeInputError(
                "Combined CSV event metadata contains an unsupported REDCap "
                "arm; outcomes support arm_1 (CHR) and arm_2 (HC)"
            )

        # A generic event identifies the timepoint but not the cohort. Finish
        # it with the same propagation used for a wholly absent event name.
        missing_event = event_arms.eq("")
        if missing_event.any():
            row_arms = pd.Series("", index=frame.index, dtype=object)
            if "chrcrit_part" in frame.columns:
                row_arms = frame["chrcrit_part"].map(_cohort_arm)
            propagated_arms = frame["subjectid"].map(resolved_arms).fillna("")
            row_arms = row_arms.mask(row_arms.eq(""), propagated_arms)
            frame.loc[missing_event, "redcap_event_name"] = [
                _event_name_for_timepoint(timepoint, arm)
                for timepoint, arm in zip(
                    frame.loc[missing_event, "_outcome_timepoint"],
                    row_arms.loc[missing_event],
                )
            ]
        for subject_id, arm in zip(
            frame["subjectid"], frame["redcap_event_name"].map(_event_arm)
        ):
            if arm:
                final_arms_by_subject.setdefault(str(subject_id), set()).add(arm)

        unresolved = frame["redcap_event_name"].eq("")
        unresolved_subjects.update(frame.loc[unresolved, "subjectid"])
        event_conflict = frame["redcap_event_name"].ne("") & frame[
            "redcap_event_name"
        ].map(_event_timepoint).ne(frame["_outcome_timepoint"])
        if event_conflict.any():
            raise OutcomeInputError(
                "One or more reconstructed REDCap events conflict with their "
                "combined-CSV timepoint"
            )

        duplicate_events = frame.duplicated(
            subset=["subjectid", "redcap_event_name"], keep=False
        )
        duplicate_count += frame.loc[
            duplicate_events, ["subjectid", "redcap_event_name"]
        ].drop_duplicates().shape[0]

        frame.drop(columns=["_outcome_timepoint"], inplace=True)
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].mask(frame[column].eq(""), np.nan)
        frame.set_index("subjectid", drop=False, inplace=True)
        prepared_slices.append(frame)

    mixed_final_arm_subjects = sum(
        len(arms) > 1 for arms in final_arms_by_subject.values()
    )
    if mixed_final_arm_subjects:
        raise OutcomeInputError(
            "Outcome calculations do not support mixed REDCap arm histories; "
            f"found {mixed_final_arm_subjects} subject(s) after event reconstruction"
        )

    if unresolved_subjects:
        raise OutcomeInputError(
            "Could not determine a REDCap arm from event metadata or "
            f"chrcrit_part for {len(unresolved_subjects)} subject(s)"
        )
    if duplicate_count:
        raise OutcomeInputError(
            f"Combined CSVs contain {duplicate_count} duplicate subject/event "
            "pair(s); outcome calculations require one logical row per event"
        )

    return CombinedSubjectStore(prepared_slices, tuple(subject_order))


def pull_data(
    combined_subjects: Mapping[str, pd.DataFrame], subject_id: str
) -> pd.DataFrame:
    """Return an isolated subject frame from the combined-CSV preload."""
    normalized_id = str(subject_id).strip()
    try:
        return combined_subjects[normalized_id].copy(deep=True)
    except KeyError:
        raise OutcomeInputError(
            "Requested subject is absent from the combined CSVs"
        ) from None
