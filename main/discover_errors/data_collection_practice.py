import pandas as pd 


class CollectDataPractice():

    def __init__(self):
        self.comb_csv_path = '/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/'
        self.output_dict = {}

    def create_timepoint_list(self):
        """
        Organizes every timepoint
        as list

        Returns
        ------------
        timepoint_list: list of timepoints
        """

        tp_list = ['screening','baseline']
        for x in range(1,13):
            tp_list.append('month'+f'{x}') 
        tp_list.append('month18')
        tp_list.append('month24')

        return tp_list

    def run_script(self):
        self.loop_timepoints() 


    def loop_timepoints(self):
        tp_list = self.create_timepoint_list()
        print(tp_list)
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                if tp not in ['baseline']:
                    continue
                csv_path = (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv')
                df = pd.read_csv(csv_path, keep_default_na = False)
                for row in df.itertuples():
                    lang_val = getattr(row,'chrdemo_firstlang')
                    subject = getattr(row,'subjectid')
                    self.output_dict.setdefault(lang_val, 0)
                    self.output_dict[lang_val] +=1 
                
        output_list =[]
        for lang_key, count in self.output_dict.items():
            output_list.append({'language': lang_key,'count':count})

        output_df = pd.DataFrame(output_list)
        output_df.to_csv('lang_vals.csv', index = False)
                    


if __name__ == '__main__':
    CollectDataPractice().run_script()