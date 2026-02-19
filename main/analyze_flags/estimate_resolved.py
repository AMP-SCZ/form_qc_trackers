import pandas as pd
import os
import sys 
import json
import numpy as np

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
print(parent_dir)
from utils.utils import Utils

class ResolvedEstimator():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depend_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        self.form_vars = self.utils.load_dependency_json('important_form_vars.json')
        self.all_results = []
        self.final_output = []
        self.forms_per_tp = self.utils.load_dependency_json('forms_per_timepoint.json')
        self.subject_info = self.utils.load_dependency_json('subject_info.json')

        self.master = pd.DataFrame()

    def run_script(self):
        self.loop_dropbox() 
    
    def loop_dropbox(self):
        dbx = self.utils.collect_dropbox_credentials()
        for network in dbx.files_list_folder(self.dropbox_path).entries:
            if network.name in ['PRESCIENT']:
                network_dir = self.dropbox_path + f'{network.name}'
                #for network_entry in dbx.files_list_folder(network_dir).entries:
                combined_output = network_dir + f'/combined/{network.name}_Output_V2.xlsx'
                self.read_dropbox_data(self.formatted_column_names[network.name]["combined"],
                ['manually_resolved','comments'], combined_output, dbx, network.name, ['Main Report'])

    def recover_old_flags(self, path):
        """function to recover history of specified row over time"""
        dbx = self.utils.collect_dropbox_credentials()
        md = dbx.files_get_metadata(path)  
        file_id = md.id                          
        rev_result = dbx.files_list_revisions(
            path=file_id,
            mode=dropbox.files.ListRevisionsMode.id,
            limit=100,  
        )

        for idx, entry in enumerate(rev_result.entries, start=1):
            if idx > 40:
                break
            rev = entry.rev
            when = entry.server_modified
            md2, resp = dbx.files_download(path=path, rev=rev)
            df = pd.read_excel(BytesIO(resp.content), keep_default_na = False)  
            #print('RECOVER COMMENTS TEST')
            #print(f"[{idx}] rev={rev}  modified={when}  shape={df.shape}")
            #print(df)
            #print(path)
            cols = ['Participant','Site Comments','Network Comments',
            'Manually Resolved','Date Resolved','Form','Flags']
            tmp = df[cols].replace(r"^\s*$", pd.NA, regex=True)
            mask_any_nonblank = tmp.notna().any(axis=1)
            df_filtered = df[mask_any_nonblank]
            self.master = pd.concat([self.master, df_filtered], ignore_index=True)
            print(self.master)
            print(path)
            print(idx)
        
        self.master = self.master.drop_duplicates(subset=["Participant",
        "Timepoint","Form","Flags"])
        self.master.to_csv('recovered_flags.csv', index = False)

    def append_recovered_comments(self, network):
        comments_df = pd.read_csv(f'{self.output_path}recovered_comments.csv',
                                keep_default_na=False)

        reversed_dict = self.utils.reverse_dictionary(
            self.formatted_column_names[network]['combined']
        )
        comments_df = comments_df.rename(
            columns={c: reversed_dict.get(c, c) for c in comments_df.columns}
        )

        key_cols = ['subject', 'displayed_timepoint', 'displayed_form']

        # keep only the columns we care about from recovered comments
        # (so we don't accidentally merge on comment columns)
        comment_cols = ['site_comments', 'network_comments', 'manually_resolved']
        comment_cols = [c for c in comment_cols if c in comments_df.columns]

        comments_df = comments_df[key_cols + comment_cols].drop_duplicates()

        # 4. merge onto the current tracker by key only
        merged = self.combined_tracker.merge(
            comments_df,
            on=key_cols,
            how='left',
            suffixes=('', '_rec')
        )

        for col in comment_cols:
            rec_col = f'{col}_rec'
            if rec_col in merged.columns:
                # treat '' as blank
                merged[col] = merged[col].where(merged[col] != '', merged[rec_col])
                merged = merged.drop(columns=[rec_col])

        merged.to_csv(self.curr_output_csv_path, index=False)

    def read_dropbox_data(self,
        col_names,columns_to_read,
        dropbox_path, dbx, network, reports_to_read,
        excl_report = True
    ):
        reversed_col_translations = self.utils.reverse_dictionary(col_names)
        print('STAGE 1')
        if self.check_dbx_file_exists(dbx, dropbox_path) == False:
            return
        print('STAGE 2')
        _, res = dbx.files_download(dropbox_path)
        data = res.content
        excel_data = pd.ExcelFile(BytesIO(data))
        sheet_names = excel_data.sheet_names
        prev_output_df = pd.read_csv(
            self.out_paths['current'],
            keep_default_na=False,
            engine="python",         
            on_bad_lines="skip",    
            quotechar='"',
            escapechar='\\',
        )
        print(reports_to_read)
        orig_columns = prev_output_df.columns
        for report in sheet_names:
            if report not in reports_to_read and excl_report == True:
                continue
            report_df = pd.read_excel(BytesIO(data),\
                sheet_name=report, keep_default_na = False)
            
            report_df.rename(columns=reversed_col_translations, inplace=True)
            report_df = report_df.assign(
                error_message=report_df['error_message'].apply(lambda x: x.split(' | '))
            )

            subjects_to_merge = report_df['subject'].tolist()
            report_df = report_df.explode('error_message').reset_index(drop=True)
            report_df['current_report'] = report
            prev_output_df['current_report'] = np.where(
            prev_output_df['reports'].str.contains(report, case=False), report, '')

            prev_output_df = pd.merge(prev_output_df,report_df, on=[
            'displayed_form','displayed_timepoint',
            'subject','error_message'],
            how = 'left',suffixes=('', '_dbx'))
            prev_output_df = prev_output_df.fillna('')
            
        for col_to_read in columns_to_read:
            dbx_col = f"{col_to_read}_dbx"

            if dbx_col not in prev_output_df.columns:
                print(f"[WARN] Expected {dbx_col} not found in merge. Skipping.")
                continue

            col_values = prev_output_df[dbx_col]

            if isinstance(col_values, pd.DataFrame):
                col_values = col_values.iloc[:, 0]

            has_new = col_values.astype(str).str.len() > 0
            prev_output_df.loc[has_new, col_to_read] = col_values[has_new]
        prev_output_df.to_csv(self.out_paths['current'], index = False)

if __name__ == '__main__':
    ResolvedEstimator.run_script()