import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils


class SofasFollowupMonth24Completed():
    """
    Standalone analysis script.

    Writes a CSV of every participant whose SOFAS follow-up form at the
    month 24 timepoint is marked complete AND whose completion happened more
    than DAYS_THRESHOLD (60) days before today.

    "Completed" = the form's completion variable equals 2 ("Complete"),
    matching how the QC pipeline gates a form as done
    (form_check.standard_form_filter). On PRESCIENT the completion status
    lives in the `_rpms` variable, same convention as the pipeline.

    "More than 60 days ago" is measured from the form's interview date
    (chrsofas_interview_date_fu) — the canonical date the pipeline uses for a
    form (important_form_vars[...]['interview_date_var']). Switch
    DATE_VAR_KEY to 'entry_date_var' if you'd rather anchor on the REDCap
    entry date instead.

    Reads only the two month-24 combined CSVs (PRONET + PRESCIENT). Missing
    files / columns are logged and skipped rather than aborting, mirroring
    the rest of the pipeline's defensive posture.
    """

    FORM = 'sofas_followup'
    TIMEPOINT = 'month24'
    DAYS_THRESHOLD = 60
    DATE_VAR_KEY = 'interview_date_var'
    OUTPUT_FILENAME = 'sofas_followup_month24_completed_over_60_days_ago.csv'

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json', 'r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.output_path = self.config_info['paths']['output_path']

        self.important_form_vars = self.utils.load_dependency_json(
            'important_form_vars.json')
        form_vars = self.important_form_vars[self.FORM]
        self.completion_var = form_vars['completion_var']       # sofas_followup_complete
        self.date_var = form_vars[self.DATE_VAR_KEY]            # chrsofas_interview_date_fu

        self.final_output_list = []

    def run_script(self):
        self.collect_completed_participants()
        self.write_output()

    def completion_var_for_network(self, network):
        # PRESCIENT records completion via the `_rpms` variable (same for
        # both cohorts). Mirrors form_check.standard_form_filter's rewrite.
        if network == 'PRESCIENT':
            return (self.completion_var + '_rpms').replace(
                '_hc', '').replace('onboarding', 'checkin')
        return self.completion_var

    def collect_completed_participants(self):
        # completion == 2 across every dtype shape (int/str/float) REDCap
        # might export it as.
        complete_codes = self.utils.all_dtype([2])
        for network in ['PRONET', 'PRESCIENT']:
            # Canonical combined-CSV path (matches qc_forms_main /
            # collect_subject_info): month24 -> month_24, PRONET -> ProNET.
            csv_path = (
                f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{self.TIMEPOINT.replace("month", "month_")}'
                f'_{network.replace("PRONET", "ProNET")}-day1to1.csv')
            try:
                combined_df = pd.read_csv(csv_path, keep_default_na=False)
            except FileNotFoundError:
                print(f"[sofas_followup_month24] combined CSV not found for {network}: "
                      f"{csv_path}. Skipping this network.")
                continue

            compl_var = self.completion_var_for_network(network)
            required = ['subjectid', compl_var, self.date_var]
            missing_cols = [c for c in required if c not in combined_df.columns]
            if missing_cols:
                print(f"[sofas_followup_month24] {network} month24 CSV missing "
                      f"column(s) {missing_cols}; skipping this network.")
                continue

            for row in combined_df.itertuples():
                if getattr(row, compl_var) not in complete_codes:
                    continue
                # Normalize slashes + strip any time component, so the same
                # value is accepted by check_if_val_date_format and parsed by
                # days_since_today (which normalizes '/' internally too).
                completed_date = str(getattr(row, self.date_var)).split(' ')[0].replace('/', '-')
                if not self.utils.check_if_val_date_format(completed_date):
                    continue
                days_ago = self.utils.days_since_today(completed_date)
                if days_ago > self.DAYS_THRESHOLD:
                    self.final_output_list.append({
                        'subjectid': row.subjectid,
                        'network': network,
                        'timepoint': self.TIMEPOINT,
                        'form': self.FORM,
                        'completed_date': completed_date,
                        'days_since_completed': days_ago,
                    })

    def write_output(self):
        # Explicit columns so the CSV always has a header row, even when no
        # participant matches.
        df = pd.DataFrame(self.final_output_list, columns=[
            'subjectid', 'network', 'timepoint', 'form',
            'completed_date', 'days_since_completed'])
        # Most-overdue first.
        df = df.sort_values('days_since_completed', ascending=False)
        out_path = f'{self.output_path}{self.OUTPUT_FILENAME}'
        df.to_csv(out_path, index=False)
        print(f"[sofas_followup_month24] wrote {len(df)} participant(s) "
              f"(SOFAS follow-up complete at month24, >{self.DAYS_THRESHOLD} days ago) "
              f"to {out_path}")


if __name__ == '__main__':
    SofasFollowupMonth24Completed().run_script()
