"""Differential parity test for roadmap item #11 (column-projected reads in
CollectSubjectInfo).

The change adds `usecols=` to each read_csv so only the needed columns are
parsed instead of the full wide CSV. This test proves that is byte-identical
to the original full-read behavior by running CollectSubjectInfo twice over
the SAME synthetic CSVs (which include noise columns the methods discard):
once with usecols (production) and once with usecols stripped (full read,
the original behavior), asserting the resulting subject_info dicts match.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _qc_harness import seed_config, project_root  # noqa: E402

seed_config()
import process_variables.collect_subject_info as csi_mod  # noqa: E402
from process_variables.collect_subject_info import CollectSubjectInfo  # noqa: E402
from utils.utils import Utils  # noqa: E402

UTILS = Utils()
_THRESHOLDS = UTILS.load_dependency_json('conversion_criteria_thresholds.json')
_PSYCHS = _THRESHOLDS['psychs']
_SCID = _THRESHOLDS['scid']

_VAR_TRANSLATIONS = {
    'chrcrit_included': {1: 'included', 0: 'excluded'},
    'chrcrit_part': {1: 'CHR', 2: 'HC'},
    'chrdemo_sexassigned': {1: 'Male', 2: 'Female'},
    'chr_statusform_screenfail': {1: 'true', 0: 'false'},
    'chr_subject_eos': {1: 'true', 0: 'false'},
}

# Every column any CollectSubjectInfo method projects to.
_NEEDED = [
    'subjectid', 'visit_status_string', 'chrcrit_part', 'chrcrit_included',
    'chrpsychs_scr_interview_date', 'chric_actigraphy', 'chric_passive',
    'chrpharm_interview_date', 'chrdemo_age_mos_chr', 'chrdemo_age_mos_hc',
    'chrdemo_age_mos2', 'chrdemo_sexassigned', 'chrdemo_interview_date',
    'chr_statusform_screenfail', 'chr_subject_eos', 'chrpharm_date_first',
    'chrpharm_date_mod', 'chrpharm_date_mod_2', 'chrconv_conv',
]
# A couple of real psychs/scid threshold vars so the criteria sweep fires.
_CRIT_VARS = (list(_PSYCHS.keys())[:2] + list(_SCID.keys())[:2])
_NOISE = [f'noise_col_{i}' for i in range(40)]  # discarded columns


def _fake_rows():
    # 3 subjects; values chosen to populate every subject_info field and to
    # trip a psychs and a scid criteria hit (value == threshold) on one.
    p0, p1 = list(_PSYCHS.keys())[:2]
    s0, s1 = list(_SCID.keys())[:2]
    base = {c: '' for c in (_NEEDED + _CRIT_VARS + _NOISE)}
    rows = []
    r1 = dict(base); r1.update({
        'subjectid': 'KC00001', 'visit_status_string': 'baseline',
        'chrcrit_part': '1', 'chrcrit_included': '1',
        'chrpsychs_scr_interview_date': '2023-03-03',
        'chric_actigraphy': '1', 'chric_passive': '2',
        'chrpharm_interview_date': '2023-03-04',
        'chrdemo_age_mos_chr': '300', 'chrdemo_sexassigned': '1',
        'chrdemo_interview_date': '2023-03-05',
        'chr_statusform_screenfail': '0', 'chr_subject_eos': '1',
        'chrpharm_date_first': '2023-04-01', 'chrconv_conv': '1',
        p0: str(_PSYCHS[p0]), s0: str(_SCID[s0]),   # criteria HITS
        'noise_col_0': 'zzz',
    })
    r2 = dict(base); r2.update({
        'subjectid': 'KC00002', 'visit_status_string': 'screening',
        'chrcrit_part': '2', 'chrcrit_included': '0',
        'chrpsychs_scr_interview_date': '2024-01-01',
        'chric_actigraphy': '', 'chric_passive': '1',
        'chrdemo_age_mos_hc': '240', 'chrdemo_sexassigned': '2',
        'chr_statusform_screenfail': '1', 'chr_subject_eos': '0',
        'chrconv_conv': '0',
        p0: '999', s0: '-9',   # missing-coded -> skipped, no hit
    })
    r3 = dict(base); r3.update({
        'subjectid': 'BM00003', 'visit_status_string': 'baseline',
        'chrcrit_part': '1', 'chrcrit_included': '1',
        'chrdemo_age_mos2': '216', 'chrdemo_sexassigned': '1',
        'chrconv_conv': '',
        p1: str(_PSYCHS[p1]), s1: str(_SCID[s1]),  # different vars hit
    })
    rows = [r1, r2, r3]
    cols = _NEEDED + _CRIT_VARS + _NOISE
    return pd.DataFrame(rows)[cols]


def _write_fixtures(d):
    df = _fake_rows()
    tps = ['screening', 'baseline', 'month_1', 'floating_forms', 'conversion']
    for net in ['ProNET', 'PRESCIENT']:
        for tp in tps:
            df.to_csv(os.path.join(
                d, f'AMPSCZ-combined-redcap_{tp}_{net}-day1to1.csv'), index=False)


def _run(comb_path):
    c = object.__new__(CollectSubjectInfo)
    c.utils = UTILS
    c.comb_csv_path = comb_path
    c.subject_info = {}
    c.var_translations = _VAR_TRANSLATIONS
    c.config_info = {'paths': {'combined_csv_path': comb_path}}
    return c()


def test_usecols_reads_byte_identical_to_full_reads(tmp_path):
    d = str(tmp_path) + os.sep
    _write_fixtures(d)

    # production: usecols projection
    new = _run(d)

    # oracle: strip usecols -> original full-read behavior
    orig_read = csi_mod.pd.read_csv

    def full_read(*args, **kwargs):
        kwargs.pop('usecols', None)
        return orig_read(*args, **kwargs)

    csi_mod.pd.read_csv = full_read
    try:
        oracle = _run(d)
    finally:
        csi_mod.pd.read_csv = orig_read

    assert new == oracle, f"subject_info diverged.\n new={new}\n oracle={oracle}"
    # sanity: non-trivial — fields populated and a criteria hit recorded
    assert new['KC00001']['cohort'] == 'CHR'
    assert new['KC00001']['converted'] is True
    assert new['KC00001']['has_psychs_criteria'] is True
    assert new['KC00001']['has_scid_criteria'] is True
    assert new['KC00002']['has_psychs_criteria'] is False  # 999 skipped
