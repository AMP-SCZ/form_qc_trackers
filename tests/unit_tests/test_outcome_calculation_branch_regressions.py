from pathlib import Path

import pandas as pd

from outcome_calculations import compute_outcomes


def _calculate(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    result = compute_outcomes(
        "ZZ00001",
        subject_data=frame,
        network_name="PRONET",
        run_version="synthetic_audit",
        output_dir=output_dir,
    )
    assert result is not None
    return result


def _baseline_values(result: pd.DataFrame, variables: list[str]) -> dict:
    selected = result[
        result["redcap_event_name"].eq("baseline_arm_1")
        & result["variable"].isin(variables)
    ]
    return dict(zip(selected["variable"], selected["value"]))


def test_missing_baseline_and_screening_do_not_create_real_pps_scores(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        [{"redcap_event_name": "month_1_arm_1", "chrcrit_part": "1"}]
    )

    result = _calculate(frame, tmp_path)

    assert _baseline_values(
        result,
        [
            "chrpps_sum1",
            "chrpps_sum6",
            "chrpps_sum7",
            "chrpps_sum8",
            "chrpps_sum9",
            "chrpps_sum10",
            "chrcssrsb_intensity_lifetime",
            "chrcssrsb_intensity_pastmonth",
        ],
    ) == {
        "chrpps_sum1": -900,
        "chrpps_sum6": -900,
        "chrpps_sum7": -900,
        "chrpps_sum8": -900,
        "chrpps_sum9": -900,
        "chrpps_sum10": -900,
        "chrcssrsb_intensity_lifetime": -900,
        "chrcssrsb_intensity_pastmonth": -900,
    }


def test_ctq_and_trait_not_applicable_remain_not_applicable(
    tmp_path: Path,
) -> None:
    ctq_columns = [
        "chrpps_lazy", "chrpps_born", "chrpps_hate", "chrpps_hurt",
        "chrpps_emoab", "chrpps_doc", "chrpps_bruise", "chrpps_belt",
        "chrpps_physab", "chrpps_beat", "chrpps_touch", "chrpps_threat",
        "chrpps_sexual", "chrpps_molest", "chrpps_sexab", "chrpps_loved",
        "chrpps_special", "chrpps_care", "chrpps_closefam",
        "chrpps_support", "chrpps_hunger", "chrpps_protect",
        "chrpps_pardrunk", "chrpps_dirty", "chrpps_docr",
    ]
    trait_columns = [
        "chrpps_taste", "chrpps_restaur", "chrpps_roller",
        "chrpps_holiday", "chrpps_tasty", "chrpps_pleasure",
        "chrpps_lookfwd", "chrpps_menu", "chrpps_actor",
        "chrpps_crackle", "chrpps_rain", "chrpps_grass", "chrpps_air",
        "chrpps_coffee", "chrpps_hair", "chrpps_yawn", "chrpps_snow",
    ]
    baseline = {
        "redcap_event_name": "baseline_arm_1",
        "chrcrit_part": "1",
        "chrdemo_age_yrs_chr": "20",
        "chrdemo_age_mos_chr": "0",
        "chrdemo_sexassigned": "1",
        "chrpps_fage": "-300",
        **{
            f"chrpps_sixmo___{index}": "-300"
            for index in [*range(1, 9), *range(10, 16)]
        },
        **{column: "-300" for column in ctq_columns + trait_columns},
    }
    frame = pd.DataFrame(
        [
            {"redcap_event_name": "screening_arm_1", "chrcrit_part": "1"},
            baseline,
            {"redcap_event_name": "month_1_arm_1", "chrcrit_part": "1"},
        ]
    )

    result = _calculate(frame, tmp_path)

    assert _baseline_values(
        result,
        ["chrpps_sum7", "chrpps_sum10", "chrpps_sum13", "chrpps_sum14"],
    ) == {
        "chrpps_sum7": -300,
        "chrpps_sum10": -300,
        "chrpps_sum13": -300,
        "chrpps_sum14": -300,
    }
