import pandas as pd


class ThresholdCheck():
    def __init__(self):
        self.scid_vars = ['chrscid_b1', 'chrscid_b2', 'chrscid_b3', 'chrscid_b4', 'chrscid_b5', 'chrscid_b6', 'chrscid_b7', 'chrscid_b8', 'chrscid_b9', 'chrscid_b10',
                          'chrscid_b11', 'chrscid_b12', 'chrscid_b13', 'chrscid_b14', 'chrscid_b16', 'chrscid_b17', 'chrscid_b18', 'chrscid_b19', 'chrscid_b20', 'chrscid_b21', 'chrscid_b24']

        self.data_dict_df = pd.read_csv('data_dictionary.csv')
        self.psychs_vars = self.data_dict_df[self.data_dict_df['Variable / Field Name'].str.contains(
            'chrpsychs', case=False, na=False)]['Variable / Field Name'].tolist()
        print(len(self.psychs_vars))

        self.rslt = []

    def loop_combined_df(self):
        tp_list = self.create_timepoint_list()
        tp_list.extend(['floating', 'conversion'])

        for network in ['PRONET', 'PRESCIENT']:
            for tp in tp_list:
                if tp == 'conversion':
                    continue

                print(tp)
                print('-------')
                combined_df = pd.read_csv(
                    f'combined_csv/combined-{network}-{tp}-day1to1.csv',
                    keep_default_na=False)

                self.check_scid_threshold(combined_df, tp)

        df_rslt = pd.DataFrame(self.rslt)
        df_rslt.to_csv('scid_threshold_checks.csv', index=False)

    def create_timepoint_list(self):
        """
        Organizes every timepoint
        as list

        Returns
        ------------
        timepoint_list: list of timepoints
        """

        tp_list = ['screening', 'baseline']
        for x in range(1, 13):
            tp_list.append('month'+f'{x}')
        tp_list.append('month18')
        tp_list.append('month24')

        return tp_list

    def check_scid_threshold(self, df, tp):
        for row in df.itertuples():
            scid_with_3 = []
            psychs_with_6 = []
            output_row = {'participant': '', 'timepoint': '',
                          'scid_vars_with_3': '', 'psychs_with_6': '', 'mismatch': ''}

            for var in self.scid_vars:
                if hasattr(row, var) and getattr(row, var) in [3, 3.0, '3', '3.0']:
                    scid_with_3.append(var)

            for var in self.psychs_vars:
                if hasattr(row, var) and getattr(row, var) in [6, 6.0, '6', '6.0']:
                    psychs_with_6.append(var)

            if row.visit_status_string == 'converted':
                continue

            if len(scid_with_3) > 0:
                output_row['participant'] = row.subjectid
                output_row['timepoint'] = tp
                output_row['scid_vars_with_3'] = '|'.join(scid_with_3)

                if len(psychs_with_6) > 0:
                    output_row['psychs_with_6'] = '|'.join(psychs_with_6)
                    output_row['mismatch'] = False
                else:
                    output_row['mismatch'] = True
                    output_row['psychs_with_6'] = 'None'

                self.rslt.append(output_row)

        self.rslt.sort(key=lambda row: not row['mismatch'])


if __name__ == '__main__':
    ThresholdCheck().loop_combined_df()
