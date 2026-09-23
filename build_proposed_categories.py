"""Produce a proposed re-categorization of a caller-supplied template workbook.

    python build_proposed_categories.py --input categorized.xlsx proposed.xlsx

Reads the source workbook read-only and writes a NEW workbook with three added
columns on the ``templates`` sheet:

  reassigned_category  -- step 1: row-level fixes, still using the original 18 names
  proposed_category    -- step 2: merged category set
  proposed_domain      -- the orthogonal subject-domain axis (see report)
  change_reason        -- why the row moved, '' when unchanged

Nothing in the pipeline consumes ``flag_category`` (grep-verified), so this is a
presentation change only. The source workbook is never modified.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent / "template_mapping_categorized (1).xlsx"


# ---------------------------------------------------------------- step 1
# Row-level reassignments, applied in order; first match wins. Each entry is
# (predicate-on-template, current_category or None for any, new_category, reason).
REASSIGN = [
    (r"days between", "Cross-Form Consistency Error", "Date/Timestamp Error",
     "visit-window interval check: a timing problem, not a field-vs-field disagreement"),
    # Only branch 1 of fluid_checks.cbc_compl_check moves. Branch 2 ("tube was NOT
    # sent ... but CBC form HAS been completed") is a genuine two-form disagreement
    # and stays put - both forms are complete, they just contradict each other.
    (r"CBC form ha(?:s|vs) not been completed", "Cross-Form Consistency Error",
     "Form Completion/Workflow Status",
     "the defect is that the CBC form was never completed (fluid_checks.py "
     "cbc_compl_check branch 1)"),
    (r"does not equal chrpharm_firstdose", "Calculated/Derived Field Error", "Date/Timestamp Error",
     "same check as 'does not match the date portion of chrpharm_firstdose_medN' "
     "(dependencies/pharm_checks_original.py:666) under an older message wording; "
     "it compares two dates and derives nothing"),
    (r"chrpps_[mf]dob", "Calculated/Derived Field Error", "Date/Timestamp Error",
     "pps_dob_age_range_check (clinical_checks_main.py:657) compares "
     "chrpps_interview_date against a parent DOB and bounds the gap at 10-85 years - "
     "two dates, no derived field"),
    (r"gf_role|gf_social|lowest score|highest score|Social Scale low score",
     "Calculated/Derived Field Error", "Cross-Variable Logical Contradiction",
     "GF low/current/high are three independently rated scores on one form that must "
     "stay ordered; an out-of-order set is a contradiction, not a miscalculation"),
    (r"source of information", "Cross-Form Consistency Error",
     "Cross-Variable Logical Contradiction",
     "tbi_info_source_check (clinical_checks_main.py:437) compares chrtbi_sourceinfo "
     "against two chrtbi_* answers on the same form - within-form, not cross-form"),
    (r"answered differently to whether", "Cross-Form Consistency Error",
     "Cross-Variable Logical Contradiction",
     "compares chrtbi_subject_head_injury against chrtbi_parent_headinjury, both on "
     "the TBI form (clinical_checks_main.py:422)"),
    (r"not been included, but blood has been collected", "Cross-Form Consistency Error",
     "Enrollment/Conversion Status Error",
     "same shape as 'Participant has been excluded, but moved beyond <tp>', already "
     "in Enrollment: study procedures performed against enrollment status"),
    (r"included or excluded", "Form Completion/Workflow Status",
     "Enrollment/Conversion Status Error",
     "the only template in Form Completion that is not '<form> not marked as "
     "complete'; it is an unset enrollment status"),
    (r"is in the future", "Out-of-Range/Implausible Value", "Date/Timestamp Error",
     "a future date is a date defect; the rest of Out-of-Range is numeric measurement bounds"),
    (r"before January", "Date Format Error", "Date/Timestamp Error",
     "an implausibly early date is not a format problem - the string parsed fine"),
    (r"improper format", "Date Format Error", "Date/Timestamp Error",
     "a required date field holding the sentinel 'NA'. Its twin - 'Date is marked "
     "with a missing code, but all interview dates are required' - is the same "
     "defect; the two are separate templates only because 'NA' is absent from "
     "utils.py:66 _MISSING_CODE_LIST. Keep them together"),
    (r"marked with a missing code", "Missing Data Code", "Date/Timestamp Error",
     "twin of the 'improper format / NA' template above: a required interview date "
     "holding a sentinel instead of a date. Actionable, unlike the rest of "
     "Missing Data Code"),
    # The two 'malformed' templates are not a defect class - they are pre-fix
    # wordings of live checks, the same situation as the firstdose row above.
    (r"found between subjects", "Uncategorized", "Duplicate/Non-Unique Identifier",
     "pre-fix wording of the cross-subject blood duplicate check now emitted at "
     "fluid_checks.py:325; the leaked tuple repr was the bug, the check is live"),
    (r"is greater', 'than the high score", "Code Artifact (malformed flag)",
     "Cross-Variable Logical Contradiction",
     "pre-fix wording of the GF high/low ordering check; belongs with the rest of "
     "that family (which this script also moves out of Calculated/Derived)"),
]

# ---------------------------------------------------------------- step 2
# Category merges, applied after step 1. Exactly one merge survived adversarial
# review; every other consolidation is expressed above as a reassignment, or in
# the report as a `maps_to` (same check, different wording).
MERGE = {
    "Duplicate Medication Course": "Medication Course Record Error",
    "Medication Course Coding Error": "Medication Course Record Error",
}

# ---------------------------------------------------------------- provenance
# Populated ONLY where the emitting check was located (or shown absent) in the
# tree. Blank means "not investigated", never "current".
PROVENANCE = [
    (r"Duplicate positions found in two different subjects",
     "RETIRED - this message exists nowhere in the tree; the position_variables "
     "group is absent from fluid_checks._PER_SUBJECT_VARS_CACHE"),
    (r"Duplicate blood (?:barcode|ID) value",
     "SUPERSEDED WORDING - fluid_checks.py:325 now emits one unified "
     "'blood ID/barcode' label for both"),
    (r"does not equal chrpharm_firstdose",
     "SUPERSEDED WORDING - reworded to 'does not match the date portion of ...' "
     "at pharm_checks_original.py:666"),
    (r"^<var> : Value is empty|^<var> : value is empty",
     "UNVERIFIED - no emitter for this string exists in the tree, and the source "
     "sheet's own maps_to asserts it equals 'Variable is blank.'. But that is a "
     "human annotation, not evidence: Pipeline_QC_Checks_Summary documents "
     "check_if_blank's message as 'Variable is blank.' even in the May-2026 "
     "revision, and this workspace has no git history (.git is a stub) to settle "
     "whether these were ever one check. Confirm before relying on it."),
    (r"found between subjects|is greater', 'than the high score",
     "SUPERSEDED WORDING - pre-fix message with a leaked Python tuple repr"),
]

# ---------------------------------------------------------------- domain axis
# Orthogonal to mechanism. Mirrors how the pipeline already routes flags to
# tracker tabs via the `reports` field (Main / Date / Blood / Fluids / Cognition /
# Scid / Missingness / Digital / Secondary Report).
DOMAIN_RULES = [
    (r"barcode|blood ID|Duplicate positions|freezer|rack|EDTA|CBC|Volume|Processing time|"
     r"blood volume|blood draw|sent to lab|time fasting|Incorrect units", "Specimen/Blood"),
    (r"pharm|medication|med\d|dosage|firstdose|lastuse|onset|offset|ongoing", "Medication"),
    (r"scid|manic|hypomanic|depress|schizophren|bipolar|MOOD EPISODES|criteria are NOT fulfilled",
     "SCID/Diagnosis"),
    (r"GUID", "Identifiers/Registry"),
    (r"IQ|FSIQ|T-Score|penncnb", "Cognition"),
    (r"not marked as complete|moved onto next timepoint|started the next timepoint|"
     r"included or excluded|converted|excluded|inclusion form", "Enrollment/Workflow"),
    (r"psychs|bprs|oasis|sofas|gf_role|gf_social|nsipr|cdss|promis", "Symptom/Clinical scales"),
    (r"demographics age|date of birth|chrpps_[mf]dob|sex was not recorded|race was not recorded",
     "Demographics"),
]


def reassign(row):
    tpl = str(row["canonical_template"])
    cat = row["flag_category"]
    for pattern, from_cat, to_cat, reason in REASSIGN:
        if from_cat is not None and cat != from_cat:
            continue
        if re.search(pattern, tpl, re.IGNORECASE):
            return to_cat, reason
    return cat, ""


def domain(row):
    tpl = str(row["canonical_template"])
    for pattern, name in DOMAIN_RULES:
        if re.search(pattern, tpl, re.IGNORECASE):
            return name
    return "General/Other"


def provenance(row):
    tpl = str(row["canonical_template"])
    for pattern, note in PROVENANCE:
        if re.search(pattern, tpl, re.IGNORECASE):
            return note
    return ""


def main(out_path, input_path=SRC):
    input_path = Path(input_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    if input_path == out_path:
        raise ValueError("Input and output workbooks must be different files.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    book = pd.ExcelFile(input_path)
    templates = book.parse("templates")
    # Drop the pasted pivot-table block that trails the real columns.
    templates = templates[[c for c in templates.columns if not c.startswith("Unnamed")]]

    reassigned = templates.apply(reassign, axis=1, result_type="expand")
    templates["reassigned_category"] = reassigned[0]
    templates["change_reason"] = reassigned[1]
    templates["proposed_category"] = templates["reassigned_category"].map(
        lambda c: MERGE.get(c, c))
    templates["proposed_domain"] = templates.apply(domain, axis=1)
    templates["provenance_note"] = templates.apply(provenance, axis=1)

    summary = (templates.groupby("proposed_category")
               .agg(n_templates=("n_entries", "size"), n_entries=("n_entries", "sum"))
               .sort_values("n_entries", ascending=False)
               .reset_index())
    summary["pct_entries"] = (100 * summary.n_entries / summary.n_entries.sum()).round(2)

    crosswalk = (templates.groupby(["flag_category", "proposed_category"])
                 .agg(n_templates=("n_entries", "size"), n_entries=("n_entries", "sum"))
                 .reset_index()
                 .sort_values(["flag_category", "n_entries"], ascending=[True, False]))

    moved = templates[templates.change_reason != ""][
        ["canonical_template", "n_entries", "networks", "flag_category",
         "proposed_category", "change_reason"]].sort_values("n_entries", ascending=False)

    stale = templates[templates.provenance_note != ""][
        ["canonical_template", "n_entries", "networks", "flag_category",
         "provenance_note"]].sort_values("n_entries", ascending=False)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="proposed_summary", index=False)
        crosswalk.to_excel(writer, sheet_name="crosswalk", index=False)
        moved.to_excel(writer, sheet_name="reassigned_rows", index=False)
        stale.to_excel(writer, sheet_name="retired_or_superseded", index=False)
        templates.to_excel(writer, sheet_name="templates", index=False)
        book.parse("variables").to_excel(writer, sheet_name="variables", index=False)

    print(f"wrote {out_path}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"{len(moved)} templates reassigned, {moved.n_entries.sum()} entries")
    print(f"categories: {templates.flag_category.nunique()} -> "
          f"{templates.proposed_category.nunique()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="categorized workbook containing templates and variables sheets")
    parser.add_argument("output", nargs="?", type=Path,
                        default=Path("template_mapping_recategorized.xlsx"),
                        help="new output workbook (default: %(default)s)")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    main(args.output, input_path=args.input)


if __name__ == "__main__":
    cli()
