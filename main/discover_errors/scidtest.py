import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


class SCIDQCChecker:
    TEST_IDS = {
        "Pronet": ['BI02450', 'CA01089', 'CM01883', 'YA16606', 'LA00834', 'LA00145', 'OR00697', 'PI01355', 'HA04408'],
        "Prescient": ['ME00772', 'ME78581', 'BM90491', 'ME33634', 'ME20845', 'BM73097', 'ME21922'],
    }

    def __init__(self, network="prescient", version="test", id_list=None, ids_file=None, output_csv="scid_qc_output.csv"):
        self.network = network.lower()
        self.Network = self.network.capitalize()
        self.version = version
        self.ids_file = ids_file or f"/data/pnl/home/gj936/U24/Clinical_qc/flowqc/REAL_DATA/{self.network}_sub_list.txt"
        self.screening_path = '/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/AMPSCZ-combined-redcap_screening_PRESCIENT-day1to1_v2.csv'
        screen_df = pd.read_csv(self.screening_path, 
        keep_default_na = False)
        self.output_csv = output_csv
        self.id_list = id_list or self._load_ids()
        self.id_list = screen_df['subjectid'].tolist()
        print(self.id_list)
        self.rows = []

    def _load_ids(self):
        if self.version in {"test", "create_control"}:
            return self.TEST_IDS[self.Network]
        ids = pd.read_csv(self.ids_file, sep="\n", header=None)[0].tolist()
        return ids if self.Network == "Pronet" else [x.split(" ")[1] if " " in x else x for x in ids[2:]]

    def _json_path(self, sid):
        site = sid[:2]
        return f"/data/predict1/data_from_nda/{self.Network}/PHOENIX/GENERAL/{self.Network}{site}/processed/{sid}/surveys/{sid}.{self.Network}.json"

    def _pull(self, sid):
        with open(self._json_path(sid)) as f:
            df = pd.DataFrame.from_dict(json.load(f), orient="columns")
        df = df.apply(lambda x: x.str.strip()).replace("", np.nan)
        return df[df["redcap_event_name"].str.contains("baseline")].fillna(-9)

    def _v(self, df, col):
        try:
            return int(df[col].astype(int).iloc[0])
        except Exception:
            return -9

    def _add(self, sid, msg):
        print(f"ID: {sid} {msg}")
        self.rows.append({"subject_id": sid, "message": msg})

    def _any_missing(self, df, *cols):
        return any(self._v(df, c) == -9 for c in cols)

    def _all_eq(self, df, val, *cols):
        return all(self._v(df, c) == val for c in cols)

    def _any_gt(self, df, val, *cols):
        return any(self._v(df, c) > val for c in cols)

    def _run_rules(self, sid, df, rules):
        for cond, msg in rules:
            if cond():
                self._add(sid, msg)

    def check_subject(self, sid, df):
        v = lambda c: self._v(df, c)

        if v("scid5_psychosis_mood_substance_abuse_complete") != 2:
            return

        rules = [
            # current depression
            (lambda: self._any_missing(df, "chrscid_a1", "chrscid_a2"),
             "one of the two main depression symptoms were not filled out. check a1, a2"),

            (lambda: v("chrscid_a1") == 1 and v("chrscid_a2") == 1 and (v("chrscid_a22_1") != -9 or v("chrscid_a22") != -9),
             "does not fulfill A1 or A2 criteria of depression, however, later information is filled out. check a22"),

            (lambda: self._all_eq(df, 3, "chrscid_a1", "chrscid_a2") and v("chrscid_a22_1") != 2,
             "does fulfill both main criteria but was counted incorrectly, check a1, a2, a22_1"),

            (lambda: ((v("chrscid_a1") == 3 and v("chrscid_a2") in (1, 2)) or (v("chrscid_a2") == 3 and v("chrscid_a1") in (1, 2))) and v("chrscid_a22_1") != 1,
             "does fulfill main criteria but further value was wrong, a1, a2, a22_1"),

            (lambda: self._any_gt(df, 1, "chrscid_a1", "chrscid_a2") and v("chrscid_a22") == -9,
             "does fulfill something from the main criteria but no further information is provided, a1, a2, a22"),

            (lambda: v("chrscid_a22") > 4 and v("chrscid_a22_1") > 0 and v("chrscid_a25") == -9,
             "subject fulfills more than 4 criteria of depression but further questions are not asked. check a22, a22_1, a25"),

            (lambda: v("chrscid_as6") > 3 and v("chrscid_as7") == -9,
             "subject fulfills current depressive episode with anxiety but severity is not filled, as6, as7"),

            # past MDD
            (lambda: v("chrscid_a27") == -9 and v("chrscid_scid5_pastmde_yes___1") == 0,
             "missing information for past major depressive disorder, a27, or pastmde box not checked"),

            (lambda: self._any_gt(df, 1, "chrscid_a27", "chrscid_a28") and v("chrscid_a48_1") == -9,
             "fulfill A1 or A2 criteria of PAST depression, however other questions are not asked, a27, a28, a48_1"),

            (lambda: self._all_eq(df, 3, "chrscid_a27", "chrscid_a28") and v("chrscid_a48_1") != 2,
             "does fulfill both main criteria but was counted incorrectly, a27, a28, a48_1"),

            (lambda: ((v("chrscid_a27") == 3 and v("chrscid_a28") in (1, 2)) or (v("chrscid_a28") == 3 and v("chrscid_a27") in (1, 2))) and v("chrscid_a48_1") != 1,
             "does fulfill main criteria but further value was wrong, a27, a28, a48_1"),

            (lambda: v("chrscid_a27") > 1 and v("chrscid_a28") > 1 and v("chrscid_a48") == -9,
             "does fulfill something from the main criteria but no further information is provided, a27, a28, a48"),

            (lambda: v("chrscid_a48") > 4 and v("chrscid_a48_1") > 0 and v("chrscid_a49") == -9,
             "subject fulfills more than 4 criteria of depression but further questions are not asked, a48, a48_1, a49"),

            # psychosis grouped checks
            (lambda: any(v(f"chrscid_b{i}") == -9 for i in range(1, 15)),
             "a symptom for delusions was not filled: check b1-b14"),

            (lambda: any(v(f"chrscid_b{i}") == -9 for i in range(16, 22)),
             "a symptom for hallucinations was not checked: check b16-b21"),

            (lambda: v("chrscid_b23___1") != 1 and any(v(f"chrscid_b{i}") == -9 for i in range(24, 40)),
             "an other symptom was not checked: check b24-b39"),

            (lambda: self._any_missing(df, "chrscid_b40", "chrscid_b42", "chrscid_b44"),
             "a negative symptom was not checked: check b40-b44"),

            (lambda: (v("chrscid_b40") == 3 and v("chrscid_b41") == -9) or (v("chrscid_b42") == 3 and v("chrscid_b43") == -9),
             "a negative symptom was present but the second question was not asked: check b40-b44"),

            # examples of skip logic
            (lambda: v("chrscid_a1") == 1 and v("chrscid_a2") == 1 and v("chrscid_a27") == -9,
             "A1 and A2 are 1 thus jump to PAST MAJOR DEPRESSIVE DISORDER but empty - a27"),

            (lambda: v("chrscid_a22") < 5 and v("chrscid_a27") == -9,
             "SXS are less than 5 (a22) thus jump to PAST MAJOR DEPRESSIVE DISORDER but empty - a27"),

            (lambda: v("chrscid_a23") == 1 and v("chrscid_a27") == -9,
             "MDD absent (a23 ==1) thus jump to PAST MAJOR DEPRESSIVE DISORDER but empty - a27"),
        ]

        self._run_rules(sid, df, rules)

        # compact repeated substance checks
        for name, yn, cols in [
            ("sedhypanx", "chrscid_sedhypanx_yn", ("chrscid_sedhypanxiol_rating___1", "chrscid_sedhypanxiol_rating___2", "chrscid_sedhypanxiol_rating___3", "chrscid_sedhypanxiol_rating___4")),
            ("cannabis", "chrscid_cannabis_yn", ("chrscid_cannabis_rating___1", "chrscid_cannabis_rating___2", "chrscid_cannabis_rating___3", "chrscid_cannabis_rating___4")),
            ("stimulants", "chrscid_stimulant_yn", ("chrscid_stimulants_rating___1", "chrscid_stimulants_rating___2", "chrscid_stimulants_rating___3", "chrscid_stimulants_rating___4")),
            ("opioids", "chrscid_opioids_yn", ("chrscid_opioids_rating___1", "chrscid_opioids_rating___2", "chrscid_opioids_rating___3", "chrscid_opioids_rating___4")),
            ("hallucinogens", "chrscid_hallucinogen_yn", ("chrscid_hallucinogens_rating___1", "chrscid_hallucinogens_rating___2", "chrscid_hallucinogens_rating___3", "chrscid_hallucinogens_rating___4")),
            ("phencyclidines", "chrscid_phencyclidine_yn", ("chrscid_phencyclidine_rating___1", "chrscid_phencyclidine_rating___2", "chrscid_phencyclidine_rating___3", "chrscid_phencyclidine_rating___4")),
            ("inhalants", "chrscid_inhalant_yn", ("chrscid_inhalants_rating___1", "chrscid_inhalants_rating___2", "chrscid_inhalants_rating___3", "chrscid_inhalants_rating___4")),
            ("othersubs", "chrscid_othersub_yn", ("chrscid_othersub_rating___1", "chrscid_othersub_rating___2", "chrscid_othersub_rating___3", "chrscid_othersub_rating___4")),
        ]:
            if v(yn) == 1 and v(cols[0]) == 0 and v(cols[1]) == 0:
                self._add(sid, f"yes to {name} but no further questions asked: {name}_rating lifetime.")
            if v(yn) == 1 and v(cols[2]) == 0 and v(cols[3]) == 0:
                self._add(sid, f"yes to {name} but no further questions asked: {name}_rating past year.")

    def run(self):
        for sid in self.id_list:
            path = self._json_path(sid)
            if not Path(path).is_file():
                self._add(sid, f"file missing: {path}")
                continue
            try:
                self.check_subject(sid, self._pull(sid))
            except Exception as e:
                self._add(sid, f"processing error: {e}")

        out = pd.DataFrame(self.rows)
        if out.empty:
            out = pd.DataFrame(columns=["subject_id", "message"])
        out.to_csv(self.output_csv, index=False)
        print(f"\nSaved {len(out)} findings to: {self.output_csv}")
        return out


if __name__ == "__main__":
    checker = SCIDQCChecker(
        network="prescient",
        version="test",
        # id_list=["BI02450"],
        output_csv="scid_qc_output_prescient.csv",
    )
    checker.run()