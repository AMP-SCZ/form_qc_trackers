"""Map study variables to clinical symptom domains by parsing the REDCap
data dictionary (or its locally available derivative).

Writes ``symptom_variable_map.xlsx`` at the project root: one row per
(domain, variable) match, with the field label, the form, which text the
match came from, and the matched terms. A variable can map to multiple
domains by design (e.g. persecutory-delusion items belong to both
``paranoia`` and ``delusions``).

Two input sources, best available wins:

1. The real data dictionary — ``{dependencies_path}data_dictionary/
   *current_data_dictionary*.csv`` (same selection rule as
   ``utils.read_data_dictionary``: lexicographically last match), or an
   explicit ``--dd <path>``. Gives Field Label + Section Header +
   Choices text and Field Type.
2. Fallback when the DD is absent (it is not part of this local slice):
   ``dependencies/grouped_variables.json`` — ``var_translations`` holds
   ``"<variable> = <text>"`` for every variable (the DD's Field Label,
   HTML-stripped, by create_variable_translations; for ~1k fields with
   an empty label the producer substitutes the raw Choices/Calculations
   text instead) and ``var_forms`` holds variable -> form. No Section
   Header / Choices / Field Type in this mode.

Matching semantics (see the README sheet of the output):

* Case-insensitive substring match of keyword stems against each text
  field (label; plus section header and choices when the real DD is the
  source), plus whole-instrument matches for forms that measure a
  single domain (e.g. oasis -> anxiety, cdss -> depression).
* Vetoes are term-level, per field: each ``redaction_veto`` phrase is
  removed from the text before keywords are matched, so 'hallucinogen'
  stops 'hallucin' without touching an independent 'perceptual' hit on
  the same label. ``context_vetoes`` kill the whole (domain, item)
  match when present anywhere in the item's text (e.g. 'peak
  intoxication' boilerplate must not count as substance use no matter
  which drug names it lists). Per-domain ``form_vetoes`` and the global
  family-history / variable vetoes work the same way.
* Nothing is silently dropped: fully vetoed (domain, variable) pairs go
  to the 'Vetoed' sheet; partially vetoed rows keep their surviving
  terms in 'Matches' with the killed terms in the vetoed_terms column.
* A match means the item is ABOUT the domain topic — it says nothing
  about whether a symptom is endorsed ('denies hallucinations' still
  matches), and severity vs presence semantics are out of scope.

Standalone on purpose: Utils() cannot construct on Windows (its
absolute_path splits realpath on '/'), so this reads config.json /
the JSONs directly instead of utils.load_dependency_json.
"""

import argparse
import json
import os
import re

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PROJECT_ROOT = "/".join(os.path.realpath(__file__).split("/")[0:-3])
OUT_XLSX = os.path.join(PROJECT_ROOT, "symptom_variable_map.xlsx")
OUT_ROWS_JSON = os.path.join(
    PROJECT_ROOT, "_audit_tmp", "symptom_variable_map_rows.json"
)

# Exact browser-export headers used across the pipeline
# (process_variables/organize_reports.py, define_important_variables.py).
DD_VAR_COL = "Variable / Field Name"
DD_FORM_COL = "Form Name"
DD_LABEL_COL = "Field Label"
DD_TYPE_COL = "Field Type"
DD_SECTION_COL = "Section Header"
DD_CHOICES_COL = "Choices, Calculations, OR Slider Labels"

# Same HTML/garbage stripping as create_variable_translations
# (process_variables/define_important_variables.py:357-368), so labels
# match grouped_variables.json var_translations text.
_HTML_TAG_RE = re.compile(r"<.*?>")
_STRIP_CHARS = ["<", ">", "/", "\n", "Â"]
# REDCap piping tokens like [chrsofas_interview_date] or
# [screening_arm_1]: variable references, not label text. Human piping
# placeholders like '[depressed mood]' contain spaces and survive.
_VAR_REF_RE = re.compile(r"\[[a-z0-9_]+\]")


def clean_label(text):
    text = _HTML_TAG_RE.sub("", str(text))
    for char in _STRIP_CHARS:
        text = text.replace(char, "")
    return text.strip()


# ---------------------------------------------------------------------------
# Symptom lexicon.
#   keywords         lowercase stems, case-insensitive substring match.
#   redaction_vetoes phrases removed from the text BEFORE keyword
#                    matching — kill only the keywords they contain
#                    (e.g. 'hallucinogen' -> 'hallucin', 'hispanic' ->
#                    'panic', 'distress' -> 'stress').
#   context_vetoes   if present anywhere in the item's text, the whole
#                    (domain, item) keyword match is vetoed.
#   form_hints       forms whose items are predominantly this domain —
#                    every non-admin variable on them maps, with
#                    matched_on='instrument form'. Multi-domain
#                    instruments (psychs, bprs, scid5) are deliberately
#                    NOT form-hinted; their items match by label.
#   form_vetoes      form-name prefixes where this domain's keyword
#                    matches are vetoed (e.g. 'irritability' probes on
#                    the TBI screen are post-concussion items).
# ---------------------------------------------------------------------------

LEXICON = {
    "delusions": {
        "keywords": [
            "delusio", "unusual thought", "thought insertion",
            "thought withdrawal", "broadcast", "grandios",
            "ideas of reference", "ideas of guilt", "magical",
            "mind reading", "read your mind", "put into your head",
            "taken out of your head", "erotomani", "nihilis",
            "overvalued idea", "non-bizarre",
            "somatic idea", "jealous idea", "religious idea",
            "under the control of", "taking special notice",
        ],
        # 'due to delusions' / 'grandiose statements' / 'rumination with
        # delusional material': BPRS do-not-rate exclusion clauses on
        # non-delusion items. 'delusional disorder, or other specified':
        # SCID differential-diagnosis boilerplate naming the diagnosis.
        # 'item 8 grandiosity' / 'p8 grandiosity': PSYCHS item-7
        # navigation targets and general-instructions methodology note.
        # 'apprehension from unusual thoughts experiences': copy-paste
        # artifact in the P9 auditory anchor tables.
        "redaction_vetoes": [
            "due to delusions", "grandiose statements",
            "rumination with delusional material",
            "delusional disorder, or other specified",
            "apprehension from unusual thoughts experiences",
            "item 8 grandiosity", "p8 grandiosity",
        ],
        "context_vetoes": [],
        "form_hints": [],
        "form_vetoes": [],
    },
    "paranoia": {
        "keywords": [
            "paranoi", "persecut", "suspici", "mistrust", "distrust",
            "being watched", "spied", "conspir", "non-bizarre",
            "taking special notice", "talking about you",
        ],
        # 'acted suspiciously': PSYCHS jealous-ideas inquiry, a delusions
        # item. 'item 2 suspiciousness': PSYCHS item-1 navigation target.
        # 'might result from suspiciousness': BPRS Unco-operativeness
        # cross-reference clause.
        "redaction_vetoes": [
            "acted suspiciously", "item 2 suspiciousness",
            "might result from suspiciousness",
        ],
        # SCID mood-congruence specifier rates delusion/mood congruence,
        # not paranoia ('themes of suspiciousness' is example content)
        "context_vetoes": ["mood-congruent psychotic features"],
        "form_hints": [],
        "form_vetoes": [],
    },
    "hallucinations_perceptual": {
        "keywords": [
            "hallucin", "perceptual", "voices", "hearing things",
            "seeing things", "illusion", "dereali", "depersonal",
            "sensations on your skin", "sensations inside",
            "smelling unpleasant", "tasted bad or strange",
        ],
        # 'item 9 ...': PSYCHS item-8 (grandiosity) navigation target
        "redaction_vetoes": [
            "hallucinogen", "item 9 auditory perceptual abnormalities",
        ],
        # SCID alcohol-withdrawal checklist ('hearing things that weren't
        # really there' as a withdrawal symptom, not a perceptual item)
        "context_vetoes": ["withdrawal sxs"],
        "form_hints": [],
        "form_vetoes": [],
    },
    "anxiety": {
        "keywords": [
            "anxi", "worr", "panic", "phobi", "fear", "nervous",
            "obsess", "compulsi", "keyed up", "on edge",
        ],
        # The last five: rater-exclusion clauses (BPRS 'do not include
        # nervous mannerisms' / 'do not infer ... from depression,
        # anxiety, or neurotic defences'), the MDE A.9 suicidality
        # parenthetical, differential-diagnosis names (SCID d37/c36),
        # and the SCID-DPQ paranoid-PD item (misspelled in the source).
        "redaction_vetoes": [
            "anxiolytic", "anti-anxiety", "nervous system", "hispanic",
            "nervous mannerisms",
            "depression, anxiety, or neurotic defences",
            "neurotic defences, anxiety or somatic complaints",
            "not just fear of dying", "mixed anxiety and depression",
            "obsessive-compulsive disorder",
            "unwarrented fear", "unwarranted fear",
        ],
        # SCID alcohol-module items list anxiety among drinking
        # consequences / withdrawal symptoms
        "context_vetoes": ["your drinking", "withdrawal sxs"],
        "form_hints": ["oasis"],
        # PSYCHS 'relationships you worried about' inquiries are
        # delusional-theme items (jealous/somatic ideas), not anxiety.
        "form_vetoes": ["psychs_"],
    },
    "depression": {
        "keywords": [
            "depress", "dysphori", "hopeless", "worthless", "guilt",
            "sadness", "melanchol", "tearful", "crying", "low mood",
            "depreciat",
        ],
        # 'delusion(s) of guilt': SCID delusion-typology bookkeeping —
        # delusions, not depression (CDSS/BPRS guilt items survive via
        # their other guilt text). 'euphoric or dysphoric' / 'not other
        # symptoms...' / 'guilt, suspiciousness...': BPRS rater-caveat
        # clauses on non-depression items. 'baby crying': PSYCHS P9
        # auditory-illusion example. 'premenstrual dysphoric': SCID
        # mania-module navigation target (PMDD section name).
        "redaction_vetoes": [
            "depressant", "delusion of guilt", "delusions of guilt",
            "euphoric or dysphoric", "not other symptoms, e.g., depression",
            "guilt, suspiciousness or grandiosity", "baby crying",
            "premenstrual dysphoric",
        ],
        # 'ideas of guilt' is the PSYCHS delusional-guilt item — it
        # belongs to delusions, not depression (context veto, because
        # its probe text 'feel guilty about' would survive redaction).
        # 'your drinking': SCID alcohol-consequence items.
        "context_vetoes": ["ideas of guilt", "your drinking"],
        "form_hints": ["cdss"],
        "form_vetoes": [],
    },
    "mania_hypomania": {
        "keywords": [
            "manic", "mania", "hypomani", "grandios", "elevated mood",
            "euphori", "expansive", "racing thoughts", "thoughts racing",
            "racing through your head", "pressured speech", "talkative",
            "flight of ideas", "distractib", "decreased need for sleep",
            "less sleep than usual", "irritab",
        ],
        # 'item 8 grandiosity' / 'p8 grandiosity': PSYCHS item-7
        # navigation targets and general-instructions note. 'euphoric or
        # dysphoric': BPRS Blunted Affect rater caveat. 'go to *current
        # manic episode*': SCID navigation target on a depression-
        # specifier instruction.
        "redaction_vetoes": [
            "erotomani", "trichotillomani", "kleptomani", "pyromani",
            "irritable bowel", "euphoric or dysphoric",
            "item 8 grandiosity", "p8 grandiosity",
            "go to *current manic episode*",
        ],
        # PMDD items measure premenstrual irritability (a depressive
        # disorder), not mania; 'hostility' kills the BPRS hostility
        # prompt's incidental irritability probe. (Bare 'menstrual'
        # would also kill genuine mania routing fields that mention
        # *PREMENSTRUAL DYSPHORIC DISORDER* — keep these two specific.)
        "context_vetoes": ["menstrual cycle", "menstrual period",
                           "hostility"],
        "form_hints": [],
        "form_vetoes": ["traumatic_brain_injury_screen"],
    },
    "suicidality_self_harm": {
        "keywords": [
            "suicid", "self-harm", "self harm", "self-injur", "self injur",
            "self-mutilat", "nssi", "wish to be dead", "better off dead",
            "thoughts of own death", "lethal",
        ],
        "redaction_vetoes": [],
        "context_vetoes": [],
        "form_hints": ["cssrs_baseline", "cssrs_followup"],
        "form_vetoes": [],
    },
    "anhedonia_avolition_negative": {
        "keywords": [
            "anhedon", "avolit", "apath", "alogia", "asocial", "amotivat",
            "blunted", "flat affect", "affective flattening",
            "emotional withdrawal", "social withdrawal", "loss of interest",
            "lack of interest", "interest or pleasure",
            "diminished interest", "diminished expression",
            "negative symptom",
        ],
        # BPRS Depression exclusion clause; SCID schizoaffective
        # criterion's definitional parenthetical ('not limited to
        # anhedonia' clarifies the required MDE, it isn't an item)
        "redaction_vetoes": [
            "amotivation that accompanies", "not limited to anhedonia",
        ],
        "context_vetoes": [],
        "form_hints": ["nsipr"],
        "form_vetoes": [],
    },
    "disorganization_ftd": {
        "keywords": [
            "disorgani", "thought disorder", "formal thought", "tangenti",
            "circumstanti", "derail", "incoheren", "loose associat",
            "loosening of associat", "neologis", "perseverat",
            "odd speech", "bizarre behavior",
        ],
        # PSYCHS item-14 navigation target; SCID catatonia 'Mannerism'
        # items use 'circumstantial' in the motor sense; BPRS do-not-rate
        # clauses on Unusual Thought Content and Distractibility.
        "redaction_vetoes": [
            "item 15 disorganized communication expression",
            "circumstantial caricature",
            "not the degree of disorganisation",
            "circumstantiality, tangentiality or flight of ideas",
        ],
        # ADHD-inattention screen ('disorganized' = everyday
        # organization); NSI-PR alogia notes whose only disorganization
        # mention is its own exclusion sentence
        "context_vetoes": ["screening for inattention",
                           "disorganization are not rated here"],
        "form_hints": [],
        # staff interviewing-technique guidance ('not to derail...')
        "form_vetoes": ["speech_sampling_run_sheet"],
    },
    "sleep": {
        "keywords": [
            "sleep", "insomni", "hypersomni", "nightmare", "drowsi",
            "somnolen", "awaken",
        ],
        "redaction_vetoes": [
            # substance / medical / suicidality contexts
            "sleeping pills", "sleep apnea", "sleep and not wake up",
            "medicine for sleep", "help you sleep",
            # PPS excitement item; SCID GMC-coding example
            "hardly sleep", "insomnia due to severe back pain",
        ],
        # First two: mania decreased-need-for-sleep items (context veto,
        # not redaction — their '(How much sleep did you get?)' follow-up
        # text survives redaction). Rest: SCID alcohol-module
        # consequence/withdrawal items and the other-substances list
        # ('OTC medications for ... sleep').
        "context_vetoes": [
            "decreased need for sleep", "less sleep than usual",
            "your drinking", "withdrawal sxs", "anabolic steroids",
        ],
        "form_hints": ["item_promis_for_sleep"],
        # cssrs: passive-ideation items phrase death wishes via sleep
        "form_vetoes": ["mri_run_sheet", "cssrs", "traumatic_brain_injury_screen"],
    },
    "substance_use": {
        "keywords": [
            "substance", "alcohol", "cannabis", "marijuana", "tobacco",
            "nicotin", "smok", "vape", "vaping", "caffein", "cocaine",
            "amphetamin", "stimulant", "opioid", "opiate", "heroin",
            "ketamine", "methadone", "pain killer", "hallucinogen",
            "inhalant", "sedative", "phencyclidine", "intoxicat",
            "drug use", "drug abuse", "street drug",
        ],
        "redaction_vetoes": [],
        # PSYCHS per-symptom exclusion boilerplate ('did X occur only
        # during peak intoxication...' / '...only in some intoxications')
        # measures psychotic symptoms, not substance use. The SCID
        # secondary-etiology rule-out family ('may be secondary',
        # 'substance-induced etiology', 'not attributable to the
        # physiological effects of a substance', the slash-stripped
        # '(SUBSTANCE/MEDICATION)' and '(using SUBSTANCE/ill...)'
        # placeholders, 'psychotic disorder due to amc') rates mood/
        # psychosis etiology, not substance use. 'isopropyl alcohol':
        # Axivity device-disinfection procedure text.
        "context_vetoes": [
            "peak intoxication", "some intoxications",
            "may be secondary", "substance-induced etiology",
            "not attributable to the physiological effects of a substance",
            "substancemedication", "using substanceill",
            "psychotic disorder due to amc", "isopropyl alcohol",
        ],
        "form_hints": ["assist"],
        # head-injury circumstance probes mention substances; pharm logs
        # list psychotropic medication classes (treatment, not use); the
        # schizotypal PD instrument's substance mentions are etiologic
        # rule-out anchors
        "form_vetoes": [
            "traumatic_brain_injury_screen",
            "past_pharmaceutical_treatment",
            "current_pharmaceutical_treatment",
            "scid5_schizotypal",
        ],
    },
    "trauma_ptsd": {
        "keywords": [
            "trauma", "ptsd", "abuse", "neglect", "flashback", "assault",
            "molest", "victimi", "hypervigilan", "bully", "bullied",
            "nightmare", "violence", "hit so hard", "hit me so hard",
            "hit with a belt", "beaten", "racially provoked",
        ],
        "redaction_vetoes": [
            "traumatic brain", "head trauma", "substance abuse",
            "alcohol abuse", "drug abuse", "drug of abuse", "drug or abuse",
            "self-neglect", "self neglect", "neglect hygiene", "neglecting",
        ],
        "context_vetoes": ["brain injury"],
        "form_hints": [],
        "form_vetoes": ["traumatic_brain_injury_screen"],
    },
    "stress": {
        "keywords": [
            "stress", "overwhelm", "coping", "life event", "discriminat",
            "hassle",
        ],
        # 'distress(ing)' severity boilerplate is not the stress domain;
        # 'relieve stress' is C-SSRS NSSI-motivation phrasing;
        # 'discriminatory intent' is BPRS persecutory-belief content;
        # 'stress of being diagnosed' is SCID GMC-etiology boilerplate;
        # 'psychosocial stressor' is GF:S anchor-table phrasing
        "redaction_vetoes": [
            "distress", "acute stress disorder", "relieve stress",
            "discriminatory intent", "stress of being diagnosed",
            "psychosocial stressor",
        ],
        # PTSD items belong to trauma_ptsd; PMDD 'overwhelmed' item
        # measures premenstrual mood, not the stress domain
        "context_vetoes": ["post-traumatic", "posttraumatic", "ptsd",
                           "menstrual period"],
        "form_hints": [
            "perceived_stress_scale", "perceived_discrimination_scale",
        ],
        "form_vetoes": ["cssrs"],
    },
    "social_functioning": {
        "keywords": [
            "social functioning", "role functioning", "global functioning",
            "sofas", "interpersonal", "occupational", "vocational",
            "employment", "academic", "isolat", "premorbid adjustment",
        ],
        # DSM criterion boilerplate on mood items (PMDD 'interpersonal
        # conflicts', atypical-features 'interpersonal rejection
        # sensitivity' / 'social or occupational impairment', severity-
        # specifier 'impairment in social or occupational functioning')
        # — these rate mood symptoms, not functioning. GF/SOFAS anchors
        # keep matching via their own multiple terms and form hints.
        "redaction_vetoes": [
            "interpersonal conflict", "interpersonal rejection sensitivity",
            "social or occupational impairment",
            "impairment in social or occupational functioning",
        ],
        # SCID mild/moderate/severe specifier definitions
        "context_vetoes": ["in excess of those required"],
        "form_hints": [
            "global_functioning_social_scale",
            "global_functioning_social_scale_followup",
            "global_functioning_role_scale",
            "global_functioning_role_scale_followup",
            "sofas_screening", "sofas_followup",
            "premorbid_adjustment_scale",
        ],
        "form_vetoes": [],
    },
}

# FIGS records per-relative diagnosis bookkeeping — family history, not
# participant symptoms. Vetoed for every domain (reported, not dropped).
GLOBAL_FORM_VETOES = {
    "family_interview_for_genetic_studies_figs":
        "family-history instrument (relatives, not participant)",
    "digital_biomarkers_mindlamp_onboarding":
        "mindLAMP app-onboarding scripts (describe the app and data "
        "collection, not participant symptoms)",
}
GLOBAL_VAR_VETOES = {
    "chrdbb_survey_question":
        "mindLAMP onboarding script listing every EMA question",
    "chrchs_bpimage": "blood-pressure measurement procedure text",
    "chrmiss_domain_table":
        "missing-data refusal-code reference table (admin bookkeeping)",
    "chrpds_instr":
        "Pubertal Development Scale instruction boilerplate",
    "chrscid_c6":
        "SCID schizophrenia six-month duration criterion; symptom words "
        "are parenthetical examples of attenuated symptoms",
}
# Admin/workflow fields are excluded from whole-instrument expansion
# (they still match if a keyword genuinely hits their label).
# 'redcap' (not 'redcap_user') also catches chroasis_redcap;
# 'redacp_user' is a variable-name typo in cssrs_followup; '_pdf$' is
# the NSI-PR reference-document attachment. (The CDSS interviewer
# header is left to the 'ask the first question as written' navigation
# phrase so it lands on Context, not fully vetoed.)
_ADMIN_VAR_RE = re.compile(
    r"dateerror|redcap|redacp_user|entry_date|missing|comment|_pdf$"
)

# Field classification: separates core symptom items from the rater-
# workflow and companion fields that share a symptom block's vocabulary
# (PSYCHS alone has ~90 fields per symptom item, most of them
# instructions, severity-anchor tables, and dates). Core items go on
# the Matches sheet; everything else goes on Context Fields.
_FIELD_CLASS_VAR_RULES = [
    # _remind: rater-reminder display fields; _prompt/prompt_/_probes/
    # bprs *_p: interviewer elicitation scripts (the scored item lives
    # in a separate paired field).
    ("instruction", re.compile(
        r"_inst(?:r|\d|_|$)|_err\d*$|_complete_|_show_|_overview"
        r"|_remind|_prompt|prompt_|_probes?$|bprs_\w+_p$")),
    ("severity anchor table", re.compile(r"_table|_anchors$")),
    # note$/desc$/dscl$ (no underscore) catch e.g. chrdlm_dim_yesno_
    # othernote and the C-SSRS *desc/*dscl description companions.
    ("free-text companion", re.compile(
        r"_note|note$|_desc|desc$|dscl$|_other$|_specify")),
    # (?<!up): 'update' fields are diagnostic-update items, not dates
    ("date field", re.compile(r"(?<!up)date|_onset")),
]
# Module-navigation / routing / validation boilerplate by label text.
_NAVIGATION_PHRASES = (
    "proceed to item", "go to item", "check here", "check yes", "skip to",
    "do not assess", "return to *", "go to *", "automatically calculated",
    "must be entered", "please fix", "check to assess", "check this box",
    "scroll down", "ask the first question as written",
    "based on interviewer's observations", "if most recent episode is",
)
# Bare free-text companion labels that lack a recognizable var suffix.
_COMPANION_LABELS = ("specify", "please specify", "please describe")
_LONG_LABEL_CHARS = 400  # rater-instruction paragraphs


def classify_field(variable, label):
    var_lower = variable.lower()
    for field_class, pattern in _FIELD_CLASS_VAR_RULES:
        if pattern.search(var_lower):
            return field_class
    label_lower = label.lower()
    label_stripped = label_lower.strip()
    if (label_stripped in _COMPANION_LABELS
            or label_stripped.startswith("describe other")):
        return "free-text companion"
    if label_stripped.startswith("date and time"):
        return "date field"
    if any(phrase in label_lower for phrase in _NAVIGATION_PHRASES):
        return "instruction"
    # Long labels are rater-instruction paragraphs — unless they ask a
    # question (C-SSRS items embed the standard definition before the
    # probe and legitimately exceed the threshold).
    if len(label) > _LONG_LABEL_CHARS and "?" not in label:
        return "instruction"
    return "core"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.json"), "r") as file:
        return json.load(file)


def find_data_dictionary(dependencies_path, explicit_path=None):
    """Replicates utils.read_data_dictionary's selection: all files
    matching 'current_data_dictionary', lexicographically last."""
    if explicit_path:
        return explicit_path
    dd_dir = os.path.join(dependencies_path, "data_dictionary")
    if not os.path.isdir(dd_dir):
        return None
    matches = sorted(
        f for f in os.listdir(dd_dir) if "current_data_dictionary" in f
    )
    if not matches:
        return None
    return os.path.join(dd_dir, matches[-1])


def items_from_data_dictionary(dd_path):
    """One item per DD row. Section Header is forward-filled within each
    form (REDCap only stores it on the row where a section starts; it
    applies to subsequent fields until the next header)."""
    df = pd.read_csv(dd_path, keep_default_na=False)
    items = []
    has_section = DD_SECTION_COL in df.columns
    section_by_form = {}
    for row in df.itertuples(index=False):
        row = dict(zip(df.columns, row))
        form = row.get(DD_FORM_COL, "")
        section = ""
        if has_section:
            if str(row.get(DD_SECTION_COL, "")).strip() != "":
                section_by_form[form] = clean_label(row[DD_SECTION_COL])
            section = section_by_form.get(form, "")
        items.append({
            "variable": row.get(DD_VAR_COL, ""),
            "form": form,
            "field_label": clean_label(row.get(DD_LABEL_COL, "")),
            "field_type": row.get(DD_TYPE_COL, ""),
            "section_header": section,
            "choices": str(row.get(DD_CHOICES_COL, "")),
        })
    return items


def items_from_grouped_variables(dependencies_path):
    """Fallback: var_translations entries are '<variable> = <text>'
    strings (Field Label, or raw Choices/Calculations text for the ~1k
    fields with an empty label); var_forms maps variable -> form.
    REDCap variable-reference tokens like [chrsofas_interview_date] are
    stripped so stems don't match inside variable names."""
    path = os.path.join(dependencies_path, "grouped_variables.json")
    with open(path, "r", encoding="utf-8") as file:
        grouped = json.load(file)
    var_forms = grouped["var_forms"]
    items = []
    for variable, translation in grouped["var_translations"].items():
        prefix = f"{variable} = "
        if translation.startswith(prefix):
            label = translation[len(prefix):]
        else:
            # Defensive: split on the first ' = ' if the prefix differs.
            label = translation.split(" = ", 1)[-1]
        items.append({
            "variable": variable,
            "form": var_forms.get(variable, ""),
            # clean_label: the ~1k raw Choices/Calculations substitutes
            # keep their HTML (<font>, <b>...) — strip it so navigation
            # phrases like 'CHECK <FONT ...>HERE</FONT>' classify
            # correctly and tags never sit inside matched text.
            "field_label": clean_label(_VAR_REF_RE.sub(" ", label)),
            "field_type": "",
            "section_header": "",
            "choices": "",
        })
    return items


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _redact(text, phrases):
    for phrase in phrases:
        text = text.replace(phrase, " ")
    return text


def match_items(items):
    matches, vetoed = [], []
    for item in items:
        texts = [
            ("field_label", item["field_label"].lower()),
            ("section_header", item["section_header"].lower()),
            ("choices", item["choices"].lower()),
        ]
        texts = [(name, text) for name, text in texts if text != ""]
        form = item["form"].lower()
        variable = item["variable"]

        for domain, spec in LEXICON.items():
            # Keyword hits per field, with term-level redaction vetoes.
            hits, vetoed_hits, hit_sources = [], [], []
            for source_name, text in texts:
                raw_found = [kw for kw in spec["keywords"] if kw in text]
                if not raw_found:
                    continue
                redacted = _redact(text, spec["redaction_vetoes"])
                kept = [kw for kw in raw_found if kw in redacted]
                if kept:
                    hits.extend(kept)
                    hit_sources.append(source_name)
                vetoed_hits.extend(kw for kw in raw_found if kw not in kept)

            # Context vetoes kill all keyword hits for this item.
            context_hits = [
                cv for cv in spec["context_vetoes"]
                if any(cv in text for _, text in texts)
            ]
            if context_hits and hits:
                vetoed_hits.extend(hits)
                hits, hit_sources = [], []

            # Whole-instrument match (admin and globally vetoed fields
            # excluded — global vetoes must also stop form_hint
            # expansion, not just keyword hits).
            form_hit = form in spec["form_hints"]
            form_admin_vetoed = form_hit and bool(
                _ADMIN_VAR_RE.search(variable.lower())
            )
            form_global_vetoed = form_hit and (
                variable in GLOBAL_VAR_VETOES or form in GLOBAL_FORM_VETOES
            )
            if form_admin_vetoed or form_global_vetoed:
                form_hit = False

            # Form-level and variable-level vetoes on keyword matches.
            veto_reasons = []
            if hits:
                if variable in GLOBAL_VAR_VETOES:
                    veto_reasons.append(
                        f"variable: {GLOBAL_VAR_VETOES[variable]}")
                if form in GLOBAL_FORM_VETOES:
                    veto_reasons.append(
                        f"form: {GLOBAL_FORM_VETOES[form]}")
                if any(form.startswith(fv) for fv in spec["form_vetoes"]):
                    veto_reasons.append(f"form veto: {item['form']}")
                if veto_reasons:
                    vetoed_hits.extend(hits)
                    hits, hit_sources = [], []
            if context_hits:
                veto_reasons.append(
                    "context: " + ", ".join(context_hits))
            if form_admin_vetoed:
                veto_reasons.append(
                    "admin field excluded from whole-instrument match")
            if form_global_vetoed:
                veto_reasons.append(
                    "global veto excluded from whole-instrument match")
            if vetoed_hits and not veto_reasons:
                veto_reasons.append("redaction veto")

            # DD mode: a hit only in the (forward-filled) section header
            # is inherited section context, not item content — route it
            # to the Context Fields sheet instead of Matches.
            section_only = hit_sources == ["section_header"]

            if form_hit:
                hit_sources.append("instrument form")

            if hits or form_hit:
                matches.append({
                    "domain": domain,
                    "variable": variable,
                    "form": item["form"],
                    "field_type": item["field_type"],
                    "field_class": (
                        "section-header inherited" if section_only
                        else classify_field(variable, item["field_label"])),
                    "field_label": item["field_label"],
                    "matched_on": ", ".join(dict.fromkeys(hit_sources)),
                    "matched_terms": (
                        ", ".join(sorted(set(hits)))
                        or "(whole instrument)"
                    ),
                    "vetoed_terms": ", ".join(sorted(set(vetoed_hits))),
                })
            elif vetoed_hits or form_admin_vetoed or form_global_vetoed:
                vetoed.append({
                    "domain": domain,
                    "variable": variable,
                    "form": item["form"],
                    "field_label": item["field_label"],
                    "would_have_matched": (
                        ", ".join(sorted(set(vetoed_hits)))
                        or "(whole instrument)"
                    ),
                    "vetoed_by": "; ".join(veto_reasons),
                })
    return matches, vetoed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
WRAP = Alignment(wrap_text=True, vertical="top")


def write_sheet(ws, columns, rows):
    for col_idx, (name, width) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    for row_idx, row in enumerate(rows, start=2):
        for col_idx, (name, _) in enumerate(columns, start=1):
            value = row.get(name, "")
            if isinstance(value, str):
                # REDCap labels can carry XML-illegal control chars
                # (Word pastes); openpyxl hard-crashes on them.
                value = ILLEGAL_CHARACTERS_RE.sub("", value)
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = WRAP
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(rows) + 1}"


def build_readme_rows(source_desc, n_items, n_matches, n_vetoed):
    return [
        ("Purpose",
         "Maps every study variable to clinical symptom domains (delusions, "
         "paranoia, anxiety, depression, hallucinations, mania, suicidality, "
         "negative symptoms, disorganization, sleep, substance use, "
         "trauma/PTSD, stress, social functioning) by keyword-matching the "
         "REDCap data dictionary's field-label text plus whole-instrument "
         "assignments for single-domain scales."),
        ("Source", source_desc),
        ("How matching works",
         "Case-insensitive substring match of lowercase keyword stems "
         "against the field label (and section header / choices text when "
         "the real data dictionary is the source). Forms that measure a "
         "single domain (oasis=anxiety, cdss=depression, cssrs=suicidality, "
         "nsipr=negative symptoms, assist=substance use, PROMIS=sleep, "
         "GF:S/GF:R/SOFAS/PAS=functioning, PSS/discrimination=stress) map "
         "all their non-admin items via 'instrument form'. Multi-domain "
         "instruments (psychs, bprs, scid5) match item-by-item only."),
        ("Matches vs Context Fields",
         "The 'Matches' sheet holds core symptom items only. Workflow and "
         "companion fields that share a symptom block's vocabulary — rater "
         "instructions (_inst, 'proceed to ITEM...', long instruction "
         "paragraphs), interviewer prompt/probe scripts (_prompt, BPRS "
         "*_p; the scored item is a separate paired field), severity-"
         "anchor tables (_table), free-text companions "
         "(_note/_desc/_specify), and date fields — are domain-tagged on "
         "the 'Context Fields' sheet instead (see the field_class "
         "column). PSYCHS alone has ~90 fields per symptom item, most of "
         "them workflow."),
        ("A match means TOPIC, not endorsement",
         "Substring matching cannot read negation or direction: 'denies "
         "hallucinations' and 'no suicidal ideation' still match their "
         "domain. A row says the item is ABOUT the domain, not that the "
         "symptom is present. Severity-scale items (psychs 0-6, oasis 0-4, "
         "bprs 1-7) and presence/threshold items (scid 1/2/3) are not "
         "distinguished here."),
        ("Multi-domain mapping is intentional",
         "One variable can appear under several domains: persecutory "
         "delusions -> paranoia + delusions; grandiosity -> delusions + "
         "mania; nightmares -> sleep + trauma; loss of interest -> negative "
         "symptoms + depression. Forcing exclusivity would discard real "
         "clinical overlap."),
        ("Vetoes (term-level)",
         "Redaction vetoes remove a phrase before keywords match, so "
         "'hallucinogen' stops 'hallucin' (substance item) without killing "
         "an independent 'perceptual' hit on the same label; likewise "
         "'antidepressant', 'anxiolytic', 'Hispanic' (contains 'panic'), "
         "'distressing' (contains 'stress'), 'irritable bowel', 'traumatic "
         "brain', 'decreased need for sleep' (mania item, not sleep). "
         "Context vetoes kill the whole item-domain match: 'peak "
         "intoxication' (PSYCHS exclusion boilerplate is not substance "
         "use), PTSD phrasing in the stress domain. Form-level vetoes: "
         "FIGS (family history about relatives, vetoed everywhere), the "
         "TBI screen (post-concussion irritability/assault items), psychs "
         "forms for the anxiety domain ('worried about' inquiries there "
         "are delusional-theme items). Nothing is silently dropped — fully "
         "vetoed pairs are on the 'Vetoed' sheet; partially vetoed rows "
         "keep surviving terms in 'Matches' with killed terms in "
         "vetoed_terms."),
        ("Checkbox variables",
         "The data dictionary lists checkbox fields once under their base "
         "name; the data CSVs carry one column per choice as "
         "<variable>___<code>. A matched checkbox base name covers all its "
         "___N data columns."),
        ("Counts",
         f"{n_items} variables scanned; {n_matches} core (domain, "
         f"variable) matches on the Matches sheet (workflow/companion "
         f"fields on Context Fields); {n_vetoed} fully vetoed."),
        ("Regenerate / use the real DD",
         "python analyze_dataset/map_symptom_variables.py [--dd <path>]. "
         "Without --dd it uses dependencies/data_dictionary/"
         "*current_data_dictionary*.csv when present (adds section-header "
         "and choices text to the match surface), else falls back to "
         "dependencies/grouped_variables.json (field labels only). Edit "
         "LEXICON in the script to tune domains/keywords/vetoes."),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dd", help="explicit path to a data dictionary CSV")
    parser.add_argument("--out", default=OUT_XLSX, help="output xlsx path")
    args = parser.parse_args()
    if args.dd and not os.path.isfile(args.dd):
        parser.error(f"--dd path does not exist or is not a file: {args.dd}")

    config = load_config()
    dependencies_path = config["paths"]["dependencies_path"]

    dd_path = find_data_dictionary(dependencies_path, args.dd)
    if dd_path:
        items = items_from_data_dictionary(dd_path)
        source_desc = (
            f"REDCap data dictionary: {dd_path} (match surface: field "
            "label + section header (forward-filled per form) + choices "
            "text)."
        )
    else:
        items = items_from_grouped_variables(dependencies_path)
        source_desc = (
            "dependencies/grouped_variables.json (var_translations = the "
            "data dictionary's Field Label column, HTML stripped — or raw "
            "Choices/Calculations text for fields with empty labels; "
            "var_forms = variable->form). The data dictionary CSV is not "
            "present in this slice — drop current_data_dictionary.csv into "
            "dependencies/data_dictionary/ and re-run to also match "
            "section headers and choice labels."
        )

    matches, vetoed = match_items(items)
    core_matches = [m for m in matches if m["field_class"] == "core"]
    context_matches = [m for m in matches if m["field_class"] != "core"]

    domain_counts = {}
    for match in core_matches:
        domain_counts[match["domain"]] = domain_counts.get(match["domain"], 0) + 1

    wb = Workbook()
    readme = wb.active
    readme.title = "README"
    readme.cell(row=1, column=1, value="Topic").font = Font(bold=True)
    readme.cell(row=1, column=2, value="Detail").font = Font(bold=True)
    readme.column_dimensions["A"].width = 32
    readme.column_dimensions["B"].width = 130
    for idx, (topic, detail) in enumerate(
        build_readme_rows(
            source_desc, len(items), len(core_matches), len(vetoed)),
        start=2,
    ):
        cell = readme.cell(row=idx, column=1, value=topic)
        cell.font = Font(bold=True)
        cell.alignment = WRAP
        readme.cell(row=idx, column=2, value=detail).alignment = WRAP

    summary_rows = []
    for domain, spec in LEXICON.items():
        domain_core = [m for m in core_matches if m["domain"] == domain]
        domain_context = [
            m for m in context_matches if m["domain"] == domain]
        forms = {}
        for match in domain_core:
            forms[match["form"]] = forms.get(match["form"], 0) + 1
        top_forms = ", ".join(
            f"{form} ({count})" for form, count in
            sorted(forms.items(), key=lambda kv: -kv[1])[:5]
        )
        summary_rows.append({
            "domain": domain,
            "n_core": len(domain_core),
            "n_context": len(domain_context),
            "keywords": ", ".join(spec["keywords"]),
            "vetoes": "; ".join(
                spec["redaction_vetoes"] + spec["context_vetoes"]
                + spec["form_vetoes"]
            ),
            "instrument_forms": ", ".join(spec["form_hints"]),
            "top_matching_forms": top_forms,
        })
    summary_ws = wb.create_sheet("Summary")
    write_sheet(summary_ws, [
        ("domain", 28), ("n_core", 10), ("n_context", 10),
        ("keywords", 60), ("vetoes", 36), ("instrument_forms", 40),
        ("top_matching_forms", 60),
    ], summary_rows)

    match_columns = [
        ("domain", 26), ("variable", 28), ("form", 38), ("field_type", 12),
        ("field_class", 20), ("field_label", 80), ("matched_on", 24),
        ("matched_terms", 30), ("vetoed_terms", 20),
    ]
    matches_ws = wb.create_sheet("Matches")
    write_sheet(matches_ws, match_columns, sorted(
        core_matches, key=lambda m: (m["domain"], m["form"], m["variable"])))

    context_ws = wb.create_sheet("Context Fields")
    write_sheet(context_ws, match_columns, sorted(
        context_matches,
        key=lambda m: (m["domain"], m["form"], m["variable"])))

    vetoed_ws = wb.create_sheet("Vetoed")
    write_sheet(vetoed_ws, [
        ("domain", 26), ("variable", 28), ("form", 38),
        ("field_label", 80), ("would_have_matched", 26), ("vetoed_by", 40),
    ], sorted(vetoed, key=lambda m: (m["domain"], m["form"], m["variable"])))

    wb.save(args.out)

    os.makedirs(os.path.dirname(OUT_ROWS_JSON), exist_ok=True)
    with open(OUT_ROWS_JSON, "w", encoding="utf-8") as file:
        json.dump({"matches": matches, "vetoed": vetoed}, file, indent=2,
                  ensure_ascii=False)

    print(f"Source: {'data dictionary at ' + dd_path if dd_path else 'grouped_variables.json fallback'}")
    print(f"Scanned {len(items)} variables; {len(core_matches)} core "
          f"matches, {len(context_matches)} context-field matches, "
          f"{len(vetoed)} fully vetoed.")
    for domain in LEXICON:
        print(f"  {domain}: {domain_counts.get(domain, 0)} core")
    print(f"Wrote {args.out} and {OUT_ROWS_JSON}")


if __name__ == "__main__":
    main()
