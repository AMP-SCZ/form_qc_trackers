"""Audited source catalog for cross-form QC consistency checks.

Importing this module performs no file reads, writes, or rule evaluation.  The
pipeline adapter in :mod:`qc_types.cross_checks` promotes only rules listed in
``ENABLED_RULE_NUMBERS``; all other prototype rules retain an explicit audit
disposition below for traceability.
"""

import re

import pandas as pd

comb_csv_path = "/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/"

timepoints = [
    "screening", "baseline", "month1", "month2", "month3", "month4",
    "month5", "month6", "month7", "month8", "month9", "month10",
    "month11", "month12", "month18", "month24", "floating", "conversion"
]

networks = ["PRONET", "PRESCIENT"]

RULES = [
    '(df["chrcrit_excl9"] == 1) & (df["chrap_total"] == 0)',
    '(df["chrcrit_excl9"] == 0) & (df["chrap_total"] > 0)',
    '(df["chrfigs_mother_age"].notna()) & (df["chrpps_mage"].notna()) & ((df["chrfigs_mother_age"] - df["chrpps_mage"]).abs() > 1)',
    '(df["chrfigs_father_age"].notna()) & (df["chrpps_fage"].notna()) & ((df["chrfigs_father_age"] - df["chrpps_fage"]).abs() > 1)',
    '((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"] == 0) & (df["chrsaliva_mar"] == 1)',
    '((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_tob"] == 0) & (df["chrsaliva_tob"] == 1)',
    
    '((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days >= 0) & ((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days < 15) & (df["chrcdss_calg1"] == 0) & (df["chrbprs_bprs_depr"] > 1)',
    '((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days >= 0) & ((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days < 15) & (df["chrcdss_calg1"] > 1) & (df["chrbprs_bprs_depr"] == 1)',
    '((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days >= 0) & ((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days < 15) & (df["chrcdss_calg8"] == 0) & (df["chrbprs_bprs_suic"] > 3)',
    '((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days >= 0) & ((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days < 15) & (df["chrcdss_calg8"] > 1) & (df["chrbprs_bprs_suic"] != -9) & (df["chrbprs_bprs_suic"] <= 2)',
    '((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days >= 0) & ((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days < 15) & (df["chrcdss_calg9"] > 0) & (df["chrbprs_bprs_depr"] == 1)',
    '((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days >= 0) & ((df["chrbprs_interview_date"] - df["chrcdss_interview_date"]).dt.days < 15) & (df["chrcdss_calg9"] == 0) & (df["chrbprs_bprs_depr"] > 2)',


    '((df["chrcssrsb_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() <= 30) & (df["chrbprs_bprs_suic"] > 3) & ((df["chrcssrsb_si2l"] == 2) | (df["chrcssrsb_css_sim2"] == 0))',
    '((df["chrcssrsb_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() <= 30) & (df["chrbprs_bprs_suic"] > 4) & ((df["chrcssrsb_si3l"] == 2) | (df["chrcssrsb_css_sim3"] == 0) | (df["chrcssrsb_si4l"] == 2) | (df["chrcssrsb_css_sim4"] == 0))',
    '((df["chrcssrsb_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() <= 30) & (df["chrbprs_bprs_suic"] > 5) & ((df["chrcssrsb_si5l"] == 2) | (df["chrcssrsb_css_sim5"] == 0))',
    '((df["chrcssrsb_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() <= 30) & (df["chrbprs_bprs_suic"] > 3) & ((df["chrcssrsb_idintsvl"] < 2) | (df["chrcssrsb_css_sipmms"] < 2))',
    '((df["chrcssrsb_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() <= 30) & (df["chrbprs_bprs_suic"] > 4) & ((df["chrcssrsb_idintsvl"] < 3) | (df["chrcssrsb_css_sipmms"] < 3))',
    '((df["chrcssrsb_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() <= 30) & (df["chrbprs_bprs_suic"] > 5) & ((df["chrcssrsb_idintsvl"] < 4) | (df["chrcssrsb_css_sipmms"] < 4))',

    '((df["chroasis_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() < 22) & (df["chrbprs_bprs_anxi"] < 4) & (df["chroasis_oasis_1"] == 4)',
    '((df["chroasis_interview_date"] - df["chrbprs_interview_date"]).dt.days.abs() < 22) & (df["chrbprs_bprs_anxi"] < 5) & ((df["chroasis_oasis_4"] == 4) | (df["chroasis_oasis_5"] == 4))',
    '((df["chroasis_interview_date"] - df["chrbprs_interview_date"]).dt.days == 0) & (df["chrbprs_bprs_anxi"] == 1) & (df["chroasis_oasis_1"] > 0)',

    '((df["chrcssrsb_interview_date"] - df["chrcdss_interview_date"]).dt.days.abs() < 15) & (df["chrcdss_calg8"] > 1) & ((df["chrcssrsb_idintsvl"] < 3) | (df["chrcssrsb_css_sipmms"] < 3))',
    '((df["chrcssrsb_interview_date"] - df["chrcdss_interview_date"]).dt.days.abs() < 15) & (df["chrcdss_calg8"] > 2) & ((df["chrcssrsb_idintsvl"] < 4) | (df["chrcssrsb_css_sipmms"] < 4))',
    '((df["chrcssrsb_interview_date"] - df["chrcdss_interview_date"]).dt.days.abs() < 15) & (df["chrcdss_calg8"] > 1) & ((df["chrcssrsb_si2l"] == 2) | (df["chrcssrsb_css_sim2"] == 0))',
    '((df["chrcssrsb_interview_date"] - df["chrcdss_interview_date"]).dt.days.abs() < 15) & (df["chrcdss_calg8"] > 2) & ((df["chrcssrsb_si4l"] == 2) | (df["chrcssrsb_css_sim4"] == 0) | (df["chrcssrsb_si5l"] == 2) | (df["chrcssrsb_css_sim5"] == 0))',
    '(df["chrcssrsb_css_sipmms"] > df["chrcssrsb_idintsvl"])',
    '((df["chrcssrsb_idintsvl"] >= 3) & (df["chrcssrsb_si2l"] == 2) & (df["chrcssrsb_si3l"] == 2) & (df["chrcssrsb_si4l"] == 2) & (df["chrcssrsb_si5l"] == 2))',
    '((df["chrcssrsb_css_sipmms"] >= 3) & (df["chrcssrsb_css_sim2"] == 0) & (df["chrcssrsb_css_sim3"] == 0) & (df["chrcssrsb_css_sim4"] == 0) & (df["chrcssrsb_css_sim5"] == 0))',

    '((df["chrpds_interview_date"] - df["chrchs_interview_date"]).dt.days < 0) & (df["chrchs_mens"] == 0) & (df["chrpds_pds_f5b_p"] == 4)',
    '((df["chrpds_interview_date"] - df["chrchs_interview_date"]).dt.days > 0) & (df["chrchs_mens"] == 1) & (df["chrpds_pds_f5b_p"] == 1)',
    '((df["chrpds_interview_date"] - df["chrchs_interview_date"]).dt.days == 0) & (df["chrchs_mens"] == 1) & (df["chrpds_pds_f5b_p"] == 1)',
    '((df["chrpds_interview_date"] - df["chrchs_interview_date"]).dt.days == 0) & (df["chrchs_mens"] == 0) & (df["chrpds_pds_f5b_p"] == 4)',

    '(df["chrfigs_mother_age"].notna()) & (df["chrpps_mage"].isna())',
    '(df["chrfigs_father_age"].notna()) & (df["chrpps_fage"].isna())',
    '(df["chrpps_mage"].notna()) & (df["chrfigs_mother_age"].isna())',
    '(df["chrpps_fage"].notna()) & (df["chrfigs_father_age"].isna())',

    '(df["chrcrit_excl9"].isna()) & (df["chrap_total"].notna())',
    '(df["chrap_total"].isna()) & (df["chrcrit_excl9"].isin([0, 1]))',

    '((df["chrap_date"] - df["chrcrit_date"]).dt.days >= 0) & (df["chrcrit_excl9"] == 1) & (df["chrap_ams"] == 0) & (df["chrap_app"] == 0) & (df["chrap_asp"] == 0) & (df["chrap_brx"] == 0) & (df["chrap_crp"] == 0) & (df["chrap_cpz"] == 0) & (df["chrap_clz"] == 0) & (df["chrap_dpl"] == 0) & (df["chrap_flh"] == 0) & (df["chrap_hpl"] == 0) & (df["chrap_ilo"] == 0) & (df["chrap_lum"] == 0) & (df["chrap_lur"] == 0) & (df["chrap_olz"] == 0) & (df["chrap_pal"] == 0) & (df["chrap_pcz"] == 0) & (df["chrap_pim"] == 0) & (df["chrap_pph"] == 0) & (df["chrap_qtp"] == 0) & (df["chrap_ris"] == 0) & (df["chrap_sul"] == 0) & (df["chrap_thi"] == 0) & (df["chrap_thz"] == 0) & (df["chrap_tpz"] == 0) & (df["chrap_zpd"] == 0)',

    # chrap_pcz → no matching med code found
    # chrap_sul → no matching med code found

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_ams"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("238").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("238").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_ams"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("238").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("238").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_app"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["305","725","547","723"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["305","725","547","723"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_app"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["305","725","547","723"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["305","725","547","723"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_asp"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["530","729"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["530","729"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_asp"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["530","729"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["530","729"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_brx"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("574").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("574").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_brx"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("574").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("574").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_crp"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("721").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("721").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_crp"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("721").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("721").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_cpz"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["256","257"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["256","257"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_cpz"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["256","257"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["256","257"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_clz"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("69").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("69").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_clz"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("69").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("69").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_dpl"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["403","727"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["403","727"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_dpl"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["403","727"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["403","727"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_flh"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["220","222"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["220","222"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_flh"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["220","222"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["220","222"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_hpl"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["122","123","124"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["122","123","124"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_hpl"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["122","123","124"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["122","123","124"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_ilo"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("532").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("532").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_ilo"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("532").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("532").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_lum"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("728").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("728").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_lum"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("728").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("728").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_lur"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("540").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("540").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_lur"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("540").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("540").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_olz"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["201","726","720"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["201","726","720"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_olz"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["201","726","720"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["201","726","720"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_pal"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["531","546","722"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["531","546","722"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_pal"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["531","546","722"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["531","546","722"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_pim"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("408").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("408").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_pim"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("408").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("408").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_pph"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["270","271","110"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["270","271","110"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_pph"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["270","271","110"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["270","271","110"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_qtp"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("235").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("235").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_qtp"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("235").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("235").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_ris"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["300","425","452"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["300","425","452"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_ris"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["300","425","452"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["300","425","452"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_thi"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("184").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("184").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_thi"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("184").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("184").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_thz"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("158").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("158").any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_thz"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).eq("158").any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).eq("158").any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_tpz"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["244","245"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["244","245"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_tpz"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["244","245"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["244","245"]).any(axis=1)))',

    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days >= 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days >= 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_zpd"] == 1) & ~(df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["304","724"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["304","724"]).any(axis=1)))',
    '(((df["chrpharm_interview_date"] - df["chrap_date"]).dt.days < 0) & ((df["chrpharm_date_first"] - df["chrap_date"]).dt.days < 0) & (df.filter(regex=r"^chrpharm_med\\d+_tp$").astype(str).eq("-1").any(axis=1)) & (df["chrap_zpd"] == 0) & (df.filter(regex=r"^chrpharm_med\\d+_name_past$").astype(str).isin(["304","724"]).any(axis=1) | df.filter(regex=r"^chrpharm_med\\d+_name$").astype(str).isin(["304","724"]).any(axis=1)))',

    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_tob"] == 1) & (df["chrassist_whoassist_use1"] == 0)',
    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_mar"] == 1) & (df["chrassist_whoassist_use3"] == 0)',
    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_often1"] == 0) & (df["chrsaliva_tob"] == 1)',
    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_often3"] == 0) & (df["chrsaliva_mar"] == 1)',
    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_tob"] == 1) & (df["chrassist_whoassist_use1"] == -9)',
    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_mar"] == 1) & (df["chrassist_whoassist_use3"] == -9)',
    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_tob"] == 1) & (df["chrassist_whoassist_often1"].isin([-3, -9]))',
    '((df["chrsaliva_interview_date"] - df["chrassist_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_mar"] == 1) & (df["chrassist_whoassist_often3"].isin([-3, -9]))',

    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use3"] == 1) & (df["chrscid_cannabis_yn"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use3"] == 0) & (df["chrscid_cannabis_yn"] == 1)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use7"] == 1) & (df["chrscid_sedhypanx_yn"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use7"] == 0) & (df["chrscid_sedhypanx_yn"] == 1)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (((df["chrassist_whoassist_use4"] == 1) | (df["chrassist_whoassist_use5"] == 1))) & (df["chrscid_stimulant_yn"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use4"] == 0) & (df["chrassist_whoassist_use5"] == 0) & (df["chrscid_stimulant_yn"] == 1)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use6"] == 1) & (df["chrscid_inhalant_yn"] == 0) & (df["chrscid_othersub_yn"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use6"] == 0) & (df["chrscid_inhalant_yn"] == 1)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use9"] == 1) & (df["chrscid_opioids_yn"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use9"] == 0) & (df["chrscid_opioids_yn"] == 1)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use8"] == 1) & (df["chrscid_hallucinogen_yn"] == 0) & (df["chrscid_phencyclidine_yn"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & ((df["chrscid_hallucinogen_yn"] == 1) | (df["chrscid_phencyclidine_yn"] == 1)) & (df["chrassist_whoassist_use8"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use10"] == 1) & (df["chrscid_othersub_yn"] == 0)',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use10"] == 0) & (df["chrassist_whoassist_use6"] == 0) & (df["chrscid_othersub_yn"] == 1)',
    
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_use3"] == -9) & (df["chrscid_cannabis_yn"].isin([0, 1]))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_use7"] == -9) & (df["chrscid_sedhypanx_yn"].isin([0, 1]))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_use6"] == -9) & (df["chrscid_inhalant_yn"].isin([0, 1]))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_use9"] == -9) & (df["chrscid_opioids_yn"].isin([0, 1]))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_use10"] == -9) & (df["chrscid_othersub_yn"].isin([0, 1]))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_use8"] == -9) & ((df["chrscid_hallucinogen_yn"].isin([0, 1])) | (df["chrscid_phencyclidine_yn"].isin([0, 1])))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_use4"] == -9) & (df["chrassist_whoassist_use5"] == -9) & (df["chrscid_stimulant_yn"].isin([0, 1]))',
    
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & ((df["chrscid_sedhypanxiol_rating___1"] == 1) | (df["chrscid_sedhypanxiol_rating___3"] == 1)) & (((df["chrassist_whoassist_often7"] == 6) & (df["chrassist_whoassist_prob7"] > 3)) | ((df["chrassist_whoassist_often7"] == 6) & (df["chrassist_whoassist_fail7"] > 4)) | ((df["chrassist_whoassist_prob7"] > 3) & (df["chrassist_whoassist_fail7"] > 4)))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & ((df["chrscid_cannabis_rating___1"] == 1) | (df["chrscid_cannabis_rating___3"] == 1)) & (((df["chrassist_whoassist_often3"] == 6) & (df["chrassist_whoassist_prob3"] > 3)) | ((df["chrassist_whoassist_often3"] == 6) & (df["chrassist_whoassist_fail3"] > 4)) | ((df["chrassist_whoassist_prob3"] > 3) & (df["chrassist_whoassist_fail3"] > 4)))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & ((df["chrscid_stimulants_rating___1"] == 1) | (df["chrscid_stimulants_rating___3"] == 1)) & ((((df["chrassist_whoassist_often4"] == 6) & (df["chrassist_whoassist_prob4"] > 3)) | ((df["chrassist_whoassist_often4"] == 6) & (df["chrassist_whoassist_fail4"] > 4)) | ((df["chrassist_whoassist_prob4"] > 3) & (df["chrassist_whoassist_fail4"] > 4))) | (((df["chrassist_whoassist_often5"] == 6) & (df["chrassist_whoassist_prob5"] > 3)) | ((df["chrassist_whoassist_often5"] == 6) & (df["chrassist_whoassist_fail5"] > 4)) | ((df["chrassist_whoassist_prob5"] > 3) & (df["chrassist_whoassist_fail5"] > 4))))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & ((df["chrscid_opioids_rating___1"] == 1) | (df["chrscid_opioids_rating___3"] == 1)) & (((df["chrassist_whoassist_often9"] == 6) & (df["chrassist_whoassist_prob9"] > 3)) | ((df["chrassist_whoassist_often9"] == 6) & (df["chrassist_whoassist_fail9"] > 4)) | ((df["chrassist_whoassist_prob9"] > 3) & (df["chrassist_whoassist_fail9"] > 4)))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (((df["chrscid_hallucinogens_rating___1"] == 1) | (df["chrscid_hallucinogens_rating___3"] == 1)) & ((df["chrscid_phencyclidine_rating___1"] == 1) | (df["chrscid_phencyclidine_rating___3"] == 1))) & (((df["chrassist_whoassist_often8"] == 6) & (df["chrassist_whoassist_prob8"] > 3)) | ((df["chrassist_whoassist_often8"] == 6) & (df["chrassist_whoassist_fail8"] > 4)) | ((df["chrassist_whoassist_prob8"] > 3) & (df["chrassist_whoassist_fail8"] > 4)))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & ((df["chrscid_inhalants_rating___1"] == 1) | (df["chrscid_inhalants_rating___3"] == 1)) & (((df["chrassist_whoassist_often6"] == 6) & (df["chrassist_whoassist_prob6"] > 3)) | ((df["chrassist_whoassist_often6"] == 6) & (df["chrassist_whoassist_fail6"] > 4)) | ((df["chrassist_whoassist_prob6"] > 3) & (df["chrassist_whoassist_fail6"] > 4)))',
    '((df["chrassist_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & ((df["chrscid_othersub_rating___1"] == 1) | (df["chrscid_othersub_rating___3"] == 1)) & (((df["chrassist_whoassist_often10"] == 6) & (df["chrassist_whoassist_prob10"] > 3)) | ((df["chrassist_whoassist_often10"] == 6) & (df["chrassist_whoassist_fail10"] > 4)) | ((df["chrassist_whoassist_prob10"] > 3) & (df["chrassist_whoassist_fail10"] > 4)))',

    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_tob"] == 1) & (df["chrassist_whoassist_use1"] == 0)',
    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_tob"] == 0) & (df["chrassist_whoassist_use1"] == 1)',
    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"] == 1) & (df["chrassist_whoassist_use3"] == 0)',
    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"] == 0) & (df["chrassist_whoassist_use3"] == 1)',
    '(df["chrassist_whoassist_often1"] > 0) & (df["chrchs_tobuse"].notna()) & ((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & ((df["chrassist_interview_date"] - df["chrchs_tobuse"]).dt.days > 30)',
    '(df["chrassist_whoassist_often3"] > 0) & (df["chrchs_mardate"].notna()) & ((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & ((df["chrassist_interview_date"] - df["chrchs_mardate"]).dt.days > 30)',
    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use1"].isin([0, 1])) & (df["chrchs_tob"].isna())',
    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrassist_whoassist_use3"].isin([0, 1])) & (df["chrchs_mar"].isna())',
    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_tob"].isin([0, 1])) & ((df["chrassist_whoassist_use1"].isin([-9])) | (df["chrassist_whoassist_use1"].isna()))',
    '((df["chrassist_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"].isin([0, 1])) & ((df["chrassist_whoassist_use3"].isin([-9])) | (df["chrassist_whoassist_use3"].isna()))',
    
    '((df["chrchs_interview_date"] - df["chrassist_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_often1"] > 0) & (df["chrchs_tobuse"].isna())',
    '((df["chrchs_interview_date"] - df["chrassist_interview_date"]).dt.days >= 0) & (df["chrassist_whoassist_often3"] > 0) & (df["chrchs_mardate"].isna())',

    '(df["chrsaliva_mar"] == 1) & (df["chrchs_mardate"].notna()) & ((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & ((df["chrsaliva_interview_date"] - df["chrchs_mardate"]).dt.days > 30)',
    '(df["chrsaliva_tob"] == 1) & (df["chrchs_tobuse"].notna()) & ((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & ((df["chrsaliva_interview_date"] - df["chrchs_tobuse"]).dt.days > 30)',
    '((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_mar"] == 1) & (df["chrchs_mar"].isna())',
    '((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_tob"] == 1) & (df["chrchs_tob"].isna())',
    '((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"].isin([0, 1])) & (df["chrsaliva_mar"].isna())',
    '((df["chrsaliva_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_tob"].isin([0, 1])) & (df["chrsaliva_tob"].isna())',

    '((df["chrscid_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"] == 0) & (df["chrscid_cannabis_yn"] == 1)',
    '((df["chrscid_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"] == 1) & (df["chrscid_cannabis_yn"] == 0)',
    '((df["chrscid_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrchs_mar"].isna()) & (df["chrscid_cannabis_yn"].isin([0, 1]))',
    '((df["chrscid_interview_date"] - df["chrchs_interview_date"]).dt.days.abs() < 1) & (df["chrscid_cannabis_yn"].isna()) & (df["chrchs_mar"].isin([0, 1]))',

    '((df["chrsaliva_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_mar"] == 1) & (df["chrscid_cannabis_yn"] == 0)',
    '((df["chrsaliva_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_mar"] == 1) & (df["chrscid_cannabis_yn"].isna())',
    '((df["chrsaliva_interview_date"] - df["chrscid_interview_date"]).dt.days.abs() < 1) & (df["chrsaliva_mar"] == 1) & (df["chrscid_cannabis_yn"] == -9)',



]
assert len(RULES) == 146

RULE_LABELS = {
    1: "Inclusion/Exclusion current antipsychotic medication variable chrcrit_excl9 = Yes, but Lifetime AP Exposure total = 0",
    2: "Inclusion/Exclusion current antipsychotic medication variable chrcrit_excl9 = No, but Lifetime AP Exposure total is positive",
    3: "FIGS mother age differs from Psychosis Polyrisk mother age by more than 1 year",
    4: "FIGS father age differs from Psychosis Polyrisk father age by more than 1 year",
    5: "Current Health Status says marijuana use = No, but Daily Activity/Saliva says marijuana used today = Yes",
    6: "Current Health Status says tobacco use = No, but Daily Activity/Saliva says tobacco used today = Yes",
    7: "CDSS depression is absent, but BPRS depression is greater than not present/very mild",
    8: "CDSS depression is moderate/severe, but BPRS depression is not present",
    9: "CDSS suicide is absent, but BPRS suicidality is moderate or higher",
    10: "CDSS suicide is moderate/severe, but BPRS suicidality is absent or very mild",
    11: "CDSS observed depression is present, but BPRS depression is not present",
    12: "CDSS observed depression is absent, but BPRS depression is mild or higher",
    13: "BPRS suicidality is moderate or higher, but CSSRS non-specific active suicidal thoughts are marked No",
    14: "BPRS suicidality is moderately severe or higher, but CSSRS active suicidal thoughts/method or intent items are marked No",
    15: "BPRS suicidality is severe or higher, but CSSRS active suicidal ideation with plan is marked No",
    16: "BPRS suicidality is moderate or higher, but CSSRS most severe ideation is below non-specific active suicidal thoughts",
    17: "BPRS suicidality is moderately severe or higher, but CSSRS most severe ideation is below active suicidal thoughts with methods",
    18: "BPRS suicidality is severe or higher, but CSSRS most severe ideation is below active ideation with intent",
    19: "BPRS anxiety is below moderate, but OASIS anxiety frequency is extreme",
    20: "BPRS anxiety is below moderately severe, but OASIS anxiety interference/avoidance is extreme",
    21: "BPRS anxiety is not present on the same day, but OASIS reports anxiety symptoms",
    22: "CDSS suicide is moderate/severe, but CSSRS most severe ideation is below active suicidal thoughts with methods",
    23: "CDSS suicide is severe, but CSSRS most severe ideation is below active ideation with intent",
    24: "CDSS suicide is moderate/severe, but CSSRS non-specific active suicidal thoughts are marked No",
    25: "CDSS suicide is severe, but CSSRS intent/plan items are marked No",
    26: "CSSRS past-month most severe ideation is greater than lifetime most severe ideation",
    27: "CSSRS lifetime most severe ideation is active ideation with methods or higher, but all lifetime active ideation items are marked No",
    28: "CSSRS past-month most severe ideation is active ideation with methods or higher, but all past-month active ideation items are marked No",
    29: "PDS was completed before CHS and reports menstruation = Yes, but later CHS reports menstruation = No",
    30: "CHS was completed before PDS and reports menstruation = Yes, but later PDS reports menstruation = No",
    31: "CHS and PDS were completed same day; CHS menstruation = Yes but PDS menstruation = No",
    32: "CHS and PDS were completed same day; CHS menstruation = No but PDS menstruation = Yes",
    33: "FIGS mother age is present, but Psychosis Polyrisk mother age is missing",
    34: "FIGS father age is present, but Psychosis Polyrisk father age is missing",
    35: "Psychosis Polyrisk mother age is present, but FIGS mother age is missing",
    36: "Psychosis Polyrisk father age is present, but FIGS father age is missing",
    37: "Inclusion/Exclusion current antipsychotic medication variable chrcrit_excl9 is missing, but Lifetime AP Exposure total is present",
    38: "Lifetime AP Exposure total is missing, but Inclusion/Exclusion chrcrit_excl9 is answered",
    39: "Inclusion/Exclusion says current antipsychotic medication = Yes, but all individual Lifetime AP medication fields are No",
    40: "Lifetime AP Exposure says Amisulpride = Yes, but pharmaceutical treatment forms do not list Amisulpride",
    41: "Pharmaceutical treatment forms list Amisulpride, but Lifetime AP Exposure says Amisulpride = No",
    42: "Lifetime AP Exposure says Aripiprazole = Yes, but pharmaceutical treatment forms do not list Aripiprazole",
    43: "Pharmaceutical treatment forms list Aripiprazole, but Lifetime AP Exposure says Aripiprazole = No",
    44: "Lifetime AP Exposure says Asenapine = Yes, but pharmaceutical treatment forms do not list Asenapine",
    45: "Pharmaceutical treatment forms list Asenapine, but Lifetime AP Exposure says Asenapine = No",
    46: "Lifetime AP Exposure says Brexpiprazole = Yes, but pharmaceutical treatment forms do not list Brexpiprazole",
    47: "Pharmaceutical treatment forms list Brexpiprazole, but Lifetime AP Exposure says Brexpiprazole = No",
    48: "Lifetime AP Exposure says Cariprazine = Yes, but pharmaceutical treatment forms do not list Cariprazine",
    49: "Pharmaceutical treatment forms list Cariprazine, but Lifetime AP Exposure says Cariprazine = No",
    50: "Lifetime AP Exposure says Chlorpromazine = Yes, but pharmaceutical treatment forms do not list Chlorpromazine",
    51: "Pharmaceutical treatment forms list Chlorpromazine, but Lifetime AP Exposure says Chlorpromazine = No",
    52: "Lifetime AP Exposure says Clozapine = Yes, but pharmaceutical treatment forms do not list Clozapine",
    53: "Pharmaceutical treatment forms list Clozapine, but Lifetime AP Exposure says Clozapine = No",
    54: "Lifetime AP Exposure says Droperidol = Yes, but pharmaceutical treatment forms do not list Droperidol",
    55: "Pharmaceutical treatment forms list Droperidol, but Lifetime AP Exposure says Droperidol = No",
    56: "Lifetime AP Exposure says Fluphenazine = Yes, but pharmaceutical treatment forms do not list Fluphenazine",
    57: "Pharmaceutical treatment forms list Fluphenazine, but Lifetime AP Exposure says Fluphenazine = No",
    58: "Lifetime AP Exposure says Haloperidol = Yes, but pharmaceutical treatment forms do not list Haloperidol",
    59: "Pharmaceutical treatment forms list Haloperidol, but Lifetime AP Exposure says Haloperidol = No",
    60: "Lifetime AP Exposure says Iloperidone = Yes, but pharmaceutical treatment forms do not list Iloperidone",
    61: "Pharmaceutical treatment forms list Iloperidone, but Lifetime AP Exposure says Iloperidone = No",
    62: "Lifetime AP Exposure says Lumateperone = Yes, but pharmaceutical treatment forms do not list Lumateperone",
    63: "Pharmaceutical treatment forms list Lumateperone, but Lifetime AP Exposure says Lumateperone = No",
    64: "Lifetime AP Exposure says Lurasidone = Yes, but pharmaceutical treatment forms do not list Lurasidone",
    65: "Pharmaceutical treatment forms list Lurasidone, but Lifetime AP Exposure says Lurasidone = No",
    66: "Lifetime AP Exposure says Olanzapine = Yes, but pharmaceutical treatment forms do not list Olanzapine",
    67: "Pharmaceutical treatment forms list Olanzapine, but Lifetime AP Exposure says Olanzapine = No",
    68: "Lifetime AP Exposure says Paliperidone = Yes, but pharmaceutical treatment forms do not list Paliperidone",
    69: "Pharmaceutical treatment forms list Paliperidone, but Lifetime AP Exposure says Paliperidone = No",
    70: "Lifetime AP Exposure says Pimozide = Yes, but pharmaceutical treatment forms do not list Pimozide",
    71: "Pharmaceutical treatment forms list Pimozide, but Lifetime AP Exposure says Pimozide = No",
    72: "Lifetime AP Exposure says Perphenazine = Yes, but pharmaceutical treatment forms do not list Perphenazine",
    73: "Pharmaceutical treatment forms list Perphenazine, but Lifetime AP Exposure says Perphenazine = No",
    74: "Lifetime AP Exposure says Quetiapine = Yes, but pharmaceutical treatment forms do not list Quetiapine",
    75: "Pharmaceutical treatment forms list Quetiapine, but Lifetime AP Exposure says Quetiapine = No",
    76: "Lifetime AP Exposure says Risperidone = Yes, but pharmaceutical treatment forms do not list Risperidone",
    77: "Pharmaceutical treatment forms list Risperidone, but Lifetime AP Exposure says Risperidone = No",
    78: "Lifetime AP Exposure says Thiothixene = Yes, but pharmaceutical treatment forms do not list Thiothixene",
    79: "Pharmaceutical treatment forms list Thiothixene, but Lifetime AP Exposure says Thiothixene = No",
    80: "Lifetime AP Exposure says Thioridazine = Yes, but pharmaceutical treatment forms do not list Thioridazine",
    81: "Pharmaceutical treatment forms list Thioridazine, but Lifetime AP Exposure says Thioridazine = No",
    82: "Lifetime AP Exposure says Trifluoperazine = Yes, but pharmaceutical treatment forms do not list Trifluoperazine",
    83: "Pharmaceutical treatment forms list Trifluoperazine, but Lifetime AP Exposure says Trifluoperazine = No",
    84: "Lifetime AP Exposure says Ziprasidone = Yes, but pharmaceutical treatment forms do not list Ziprasidone",
    85: "Pharmaceutical treatment forms list Ziprasidone, but Lifetime AP Exposure says Ziprasidone = No",
    86: "Daily Activity/Saliva says tobacco used today, but ASSIST lifetime tobacco use = No",
    87: "Daily Activity/Saliva says marijuana used today, but ASSIST lifetime cannabis use = No",
    88: "Daily Activity/Saliva says tobacco used today, but ASSIST past-month tobacco frequency = Never",
    89: "Daily Activity/Saliva says marijuana used today, but ASSIST past-month cannabis frequency = Never",
    90: "Daily Activity/Saliva says tobacco used today, but ASSIST lifetime tobacco use is missing/unknown",
    91: "Daily Activity/Saliva says marijuana used today, but ASSIST lifetime cannabis use is missing/unknown",
    92: "Daily Activity/Saliva says tobacco used today, but ASSIST past-month tobacco frequency is missing/not applicable",
    93: "Daily Activity/Saliva says marijuana used today, but ASSIST past-month cannabis frequency is missing/not applicable",
    94: "ASSIST lifetime cannabis use = Yes, but SCID cannabis use = No",
    95: "ASSIST lifetime cannabis use = No, but SCID cannabis use = Yes",
    96: "ASSIST lifetime sedative/sleeping pill use = Yes, but SCID sedative/hypnotic/anxiolytic use = No",
    97: "ASSIST lifetime sedative/sleeping pill use = No, but SCID sedative/hypnotic/anxiolytic use = Yes",
    98: "ASSIST lifetime cocaine or amphetamine use = Yes, but SCID stimulant use = No",
    99: "ASSIST lifetime cocaine and amphetamine use = No, but SCID stimulant use = Yes",
    100: "ASSIST lifetime inhalant use = Yes, but SCID inhalant and other-substance use = No",
    101: "ASSIST lifetime inhalant use = No, but SCID inhalant use = Yes",
    102: "ASSIST lifetime opioid use = Yes, but SCID opioid use = No",
    103: "ASSIST lifetime opioid use = No, but SCID opioid use = Yes",
    104: "ASSIST lifetime hallucinogen use = Yes, but SCID hallucinogen and PCP/ketamine use = No",
    105: "SCID hallucinogen or PCP/ketamine use = Yes, but ASSIST lifetime hallucinogen use = No",
    106: "ASSIST lifetime other substance use = Yes, but SCID other substance use = No",
    107: "ASSIST lifetime inhalant and other-substance use = No, but SCID other substance use = Yes",
    108: "ASSIST lifetime cannabis use is missing/unknown, but SCID cannabis use is answered",
    109: "ASSIST lifetime sedative/sleeping pill use is missing/unknown, but SCID sedative/hypnotic/anxiolytic use is answered",
    110: "ASSIST lifetime inhalant use is missing/unknown, but SCID inhalant use is answered",
    111: "ASSIST lifetime opioid use is missing/unknown, but SCID opioid use is answered",
    112: "ASSIST lifetime other substance use is missing/unknown, but SCID other substance use is answered",
    113: "ASSIST lifetime hallucinogen use is missing/unknown, but SCID hallucinogen or PCP/ketamine use is answered",
    114: "ASSIST lifetime cocaine and amphetamine use are both missing/unknown, but SCID stimulant use is answered",
    115: "SCID sedative lifetime/past-year rating is 1 (absent), but ASSIST past-month responses suggest frequent/problematic use",
    116: "SCID cannabis lifetime/past-year rating is 1 (absent), but ASSIST past-month responses suggest frequent/problematic use",
    117: "SCID stimulant lifetime/past-year rating is 1 (absent), but ASSIST past-month cocaine/amphetamine responses suggest frequent/problematic use",
    118: "SCID opioid lifetime/past-year rating is 1 (absent), but ASSIST past-month responses suggest frequent/problematic use",
    119: "SCID hallucinogen and PCP lifetime/past-year ratings are 1 (absent), but ASSIST past-month responses suggest frequent/problematic use",
    120: "SCID inhalant lifetime/past-year rating is 1 (absent), but ASSIST past-month responses suggest frequent/problematic use",
    121: "SCID other-substance lifetime/past-year rating is 1 (absent), but ASSIST past-month responses suggest frequent/problematic use",
    122: "Current Health Status says tobacco use = Yes, but ASSIST lifetime tobacco use = No",
    123: "Current Health Status says tobacco use = No, but ASSIST lifetime tobacco use = Yes",
    124: "Current Health Status says marijuana use = Yes, but ASSIST lifetime cannabis use = No",
    125: "Current Health Status says marijuana use = No, but ASSIST lifetime cannabis use = Yes",
    126: "ASSIST reports tobacco use in the past month, but CHS last tobacco use date is more than 30 days before ASSIST interview",
    127: "ASSIST reports cannabis use in the past month, but CHS last marijuana use date is more than 30 days before ASSIST interview",
    128: "ASSIST lifetime tobacco use is answered, but CHS tobacco use is missing",
    129: "ASSIST lifetime cannabis use is answered, but CHS marijuana use is missing",
    130: "CHS tobacco use is answered, but ASSIST lifetime tobacco use is missing/unknown",
    131: "CHS marijuana use is answered, but ASSIST lifetime cannabis use is missing/unknown",
    132: "ASSIST reports tobacco use in the past month, but CHS last tobacco use date is missing",
    133: "ASSIST reports cannabis use in the past month, but CHS last marijuana use date is missing",
    134: "Daily Activity/Saliva says marijuana used today, but CHS last marijuana use date is more than 30 days earlier",
    135: "Daily Activity/Saliva says tobacco used today, but CHS last tobacco use date is more than 30 days earlier",
    136: "Daily Activity/Saliva says marijuana used today, but CHS marijuana use is missing",
    137: "Daily Activity/Saliva says tobacco used today, but CHS tobacco use is missing",
    138: "CHS marijuana use is answered, but Daily Activity/Saliva marijuana use today is missing",
    139: "CHS tobacco use is answered, but Daily Activity/Saliva tobacco use today is missing",
    140: "CHS marijuana use = No, but SCID lifetime cannabis use = Yes",
    141: "CHS marijuana use = Yes, but SCID lifetime cannabis use = No",
    142: "CHS marijuana use is missing, but SCID lifetime cannabis use is answered",
    143: "SCID lifetime cannabis use is missing, but CHS marijuana use is answered",
    144: "Daily Activity/Saliva says marijuana used today, but SCID lifetime cannabis use = No",
    145: "Daily Activity/Saliva says marijuana used today, but SCID lifetime cannabis use is missing",
    146: "Daily Activity/Saliva says marijuana used today, but SCID lifetime cannabis use is unknown/missing code",
}

# Only rules that survived the clinical/logic audit are promoted into the
# operational tracker.  The remainder stay in this catalog for traceability,
# but are not silently presented as errors.  Cross-instrument comparisons are
# review findings, not automatic NDA exclusions.
ENABLED_RULE_NUMBERS = frozenset({
    5, 6, 9, 10,
    29, 30, 31, 32,
    39,
    86, 88,
    94, 96, 98, 100, 101, 102, 104, 106,
    122,
    126, 127,
    134, 135,
    141, 144,
})

# Most rules require every operand to be observed.  Rule 98 is intentionally
# different: SCID combines cocaine and amphetamine into one stimulant field,
# so a Yes in either ASSIST field is sufficient evidence even if the other
# ASSIST field is missing.  The adapter consumes this metadata when building
# its observation-validity mask.
ALTERNATIVE_EVIDENCE_GROUPS = {
    98: (("chrassist_whoassist_use4", "chrassist_whoassist_use5"),),
}

# Concise dispositions for the rule families excluded from production.  This
# prevents a future maintainer from re-enabling a known false-positive family
# merely because it remains present in the source catalog.
DISABLED_RULE_REASONS = {
    1: "Duplicates rule 39 without the necessary assessment-date ordering.",
    2: "Current No and lifetime exposure Yes is valid for former users.",
    3: "FIGS age-at-death and PPS current-equivalent parental age differ semantically.",
    4: "FIGS age-at-death and PPS current-equivalent parental age differ semantically.",
    7: "Low-threshold cross-instrument depression discordance is too nonspecific.",
    8: "Cross-instrument depression discordance requires clinical validation.",
    11: "Observed versus reported depression is not a hard contradiction.",
    12: "Observed versus reported depression is not a hard contradiction.",
    13: "BPRS/C-SSRS cross-scale mapping requires clinical validation.",
    14: "BPRS/C-SSRS cross-scale mapping requires clinical validation.",
    15: "BPRS/C-SSRS cross-scale mapping requires clinical validation.",
    16: "BPRS/C-SSRS cross-scale mapping requires clinical validation.",
    17: "BPRS/C-SSRS cross-scale mapping requires clinical validation.",
    18: "BPRS/C-SSRS cross-scale mapping requires clinical validation.",
    19: "BPRS/OASIS cross-scale mapping requires clinical validation.",
    20: "BPRS/OASIS cross-scale mapping requires clinical validation.",
    21: "Any OASIS symptom versus BPRS Not Present is overly sensitive.",
    22: "CDSS/C-SSRS cross-scale mapping requires clinical validation.",
    23: "CDSS/C-SSRS cross-scale mapping requires clinical validation.",
    24: "CDSS/C-SSRS cross-scale mapping requires clinical validation.",
    25: "CDSS/C-SSRS cross-scale mapping requires clinical validation.",
    26: "Already implemented by ClinicalChecksMain.cssrs_greater_vals_check.",
    27: "Downstream C-SSRS items are legitimately blank under branching logic.",
    28: "Downstream C-SSRS items are legitimately blank under branching logic.",
    33: "Standalone cross-form missingness ignores scheduling/applicability.",
    34: "Standalone cross-form missingness ignores scheduling/applicability.",
    35: "Standalone cross-form missingness ignores scheduling/applicability.",
    36: "Standalone cross-form missingness ignores scheduling/applicability.",
    37: "Missingness is handled by the branching-aware missingness engine.",
    38: "Missingness is handled by the branching-aware missingness engine.",
    40: "Rules 40-85 duplicate the guarded MED-QC-19 pharmaceutical check.",
    87: "Saliva may reflect medical cannabis; ASSIST asks non-medical use only.",
    89: "Saliva may reflect medical cannabis; ASSIST asks non-medical use only.",
    90: "Missingness is handled by the branching-aware missingness engine.",
    91: "Missingness is handled by the branching-aware missingness engine.",
    92: "Missingness is handled by the branching-aware missingness engine.",
    93: "Missingness is handled by the branching-aware missingness engine.",
    95: "SCID Yes may reflect medical cannabis while ASSIST asks non-medical use only.",
    97: "SCID Yes may reflect prescribed sedatives while ASSIST asks non-medical use only.",
    99: "SCID Yes may reflect prescribed stimulants while ASSIST asks non-medical use only.",
    103: "SCID Yes may reflect prescribed opioids while ASSIST asks non-medical use only.",
    105: "SCID Yes may reflect medically administered ketamine while ASSIST asks non-medical use only.",
    107: "SCID Other includes medicines/OTC exposure while ASSIST asks non-medical use only.",
    108: "Missingness is handled by the branching-aware missingness engine.",
    109: "Missingness is handled by the branching-aware missingness engine.",
    110: "Missingness is handled by the branching-aware missingness engine.",
    111: "Missingness is handled by the branching-aware missingness engine.",
    112: "Missingness is handled by the branching-aware missingness engine.",
    113: "Missingness is handled by the branching-aware missingness engine.",
    114: "Missingness is handled by the branching-aware missingness engine.",
    115: "Undocumented two-of-three ASSIST heuristic needs clinical validation.",
    116: "Undocumented two-of-three ASSIST heuristic needs clinical validation.",
    117: "Undocumented two-of-three ASSIST heuristic needs clinical validation.",
    118: "Undocumented two-of-three ASSIST heuristic needs clinical validation.",
    119: "Undocumented two-of-three ASSIST heuristic needs clinical validation.",
    120: "Undocumented two-of-three ASSIST heuristic needs clinical validation.",
    121: "Undocumented two-of-three ASSIST heuristic needs clinical validation.",
    123: "Current No and lifetime Yes is valid for former users.",
    124: "CHS may reflect medical cannabis while ASSIST asks non-medical use only.",
    125: "Current No and lifetime Yes is valid for former users.",
    128: "Missingness is handled by the branching-aware missingness engine.",
    129: "Missingness is handled by the branching-aware missingness engine.",
    130: "Missingness is handled by the branching-aware missingness engine.",
    131: "Missingness is handled by the branching-aware missingness engine.",
    132: "Missingness is handled by the branching-aware missingness engine.",
    133: "Missingness is handled by the branching-aware missingness engine.",
    136: "Missingness is handled by the branching-aware missingness engine.",
    137: "Missingness is handled by the branching-aware missingness engine.",
    138: "Missingness is handled by the branching-aware missingness engine.",
    139: "Missingness is handled by the branching-aware missingness engine.",
    140: "Current No and lifetime Yes is valid for former users.",
    142: "Missingness is handled by the branching-aware missingness engine.",
    143: "Missingness is handled by the branching-aware missingness engine.",
    145: "Missingness is handled by the branching-aware missingness engine.",
    146: "Current SCID dictionary has no -9 choice; overlaps blank missingness.",
}
for _rule_number in range(40, 86):
    DISABLED_RULE_REASONS.setdefault(
        _rule_number,
        "Rules 40-85 duplicate the guarded MED-QC-19 pharmaceutical check.")

assert set(RULE_LABELS) == set(range(1, len(RULES) + 1))
assert ENABLED_RULE_NUMBERS.isdisjoint(DISABLED_RULE_REASONS)
assert ENABLED_RULE_NUMBERS | set(DISABLED_RULE_REASONS) == set(RULE_LABELS)
assert set(ALTERNATIVE_EVIDENCE_GROUPS) <= ENABLED_RULE_NUMBERS
assert all(
    set(group) <= set(re.findall(
        r'df\["([^"]+)"\]', RULES[number - 1]))
    for number, groups in ALTERNATIVE_EVIDENCE_GROUPS.items()
    for group in groups
)


DATE_COLUMNS = frozenset({
    "chrap_date",
    "chrassist_interview_date",
    "chrbprs_interview_date",
    "chrcdss_interview_date",
    "chrchs_interview_date",
    "chrchs_mardate",
    "chrchs_tobuse",
    "chrcrit_date",
    "chrcssrsb_interview_date",
    "chroasis_interview_date",
    "chrpds_interview_date",
    "chrpharm_date_first",
    "chrpharm_interview_date",
    "chrscid_interview_date",
    "chrsaliva_interview_date",
})

DATE_MISSING_CODES = frozenset({
    "", "-3", "-9", "-99", "999",
    "1901-01-01", "1903-03-03", "1909-09-09",
})


def build_file_path(tp: str, net: str, base_path: str = comb_csv_path) -> str:
    """Build a combined-export path using the pipeline's exact ProNET case."""
    tp_formatted = tp.replace("month", "month_").replace(
        "floating", "floating_forms")
    network = "ProNET" if str(net).upper() == "PRONET" else str(net)
    return (
        f"{base_path}AMPSCZ-combined-redcap_{tp_formatted}_"
        f"{network}-day1to1.csv")


def extract_columns(rule_text: str) -> list[str]:
    """Return directly referenced columns in stable expression order."""
    return list(dict.fromkeys(re.findall(r'df\["([^"]+)"\]', rule_text)))


def guess_date_columns(rule_text: str, cols: list[str]) -> list[str]:
    """Compatibility wrapper backed by explicit audited date metadata."""
    return [col for col in cols if col in DATE_COLUMNS]


def prepare_dataframe(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | set[str],
) -> pd.DataFrame:
    """Normalize rule inputs once without treating sentinels as real dates.

    All dates are normalized to calendar midnight.  This makes same-calendar
    comparisons symmetric when one REDCap instrument stores a date and another
    stores a datetime; pandas' raw ``Timedelta.dt.days`` floors negative
    partial days and previously both missed same-day pairs and admitted some
    adjacent-day pairs.
    """
    selected = [
        column for column in dict.fromkeys(columns)
        if column in df.columns]
    # Combined exports contain tens of thousands of fields.  Copy only the
    # operands used by the applicable catalog instead of duplicating the full
    # network/timepoint dataframe in memory.
    out = df.loc[:, selected].copy()
    for column in selected:
        if column in DATE_COLUMNS:
            raw = out[column].astype(str).str.strip()
            # Combined exports may serialize a REDCap sentinel either as a
            # date (``1903-03-03``) or a datetime
            # (``1903-03-03 00:00:00``).  Compare both the complete token and
            # its ISO date prefix so a missing sentinel cannot masquerade as
            # a century-old last-use date.
            raw = raw.mask(
                raw.isin(DATE_MISSING_CODES)
                | raw.str[:10].isin(DATE_MISSING_CODES))
            out[column] = pd.to_datetime(raw, errors="coerce").dt.normalize()
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def prep_df_for_rule(df: pd.DataFrame, rule_text: str) -> pd.DataFrame:
    """Prepare one rule's direct inputs (kept for standalone diagnostics)."""
    return prepare_dataframe(df, extract_columns(rule_text))


def evaluate_rule_mask(
    prepared_df: pd.DataFrame,
    rule_number: int,
) -> pd.Series:
    """Evaluate one trusted catalog expression with builtins unavailable."""
    if rule_number not in RULE_LABELS:
        raise KeyError(f"Unknown cross-check rule number: {rule_number}")
    rule_text = RULES[rule_number - 1]
    missing = [
        column for column in extract_columns(rule_text)
        if column not in prepared_df.columns]
    if missing:
        raise KeyError(
            f"Rule {rule_number} is missing required column(s): {missing}")
    mask = eval(
        compile(rule_text, f"<cross-check-{rule_number:03d}>", "eval"),
        {"pd": pd, "__builtins__": {}},
        {"df": prepared_df},
    )
    if not isinstance(mask, pd.Series):
        raise TypeError(
            f"Rule {rule_number} did not return a pandas Series")
    return mask.fillna(False).astype(bool)
