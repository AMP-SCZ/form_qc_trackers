import pandas as pd

from datetime import datetime 

new_df = pd.read_csv('/home/ob001/refactored_qc/form_qc_trackers/main/analyze_flags/recovered_flags_new_pronet.csv')
orig_df = pd.read_csv('/home/ob001/refactored_qc/form_qc_trackers/main/analyze_flags/recovered_flags_orig_pronet.csv')
df = pd.concat([new_df, orig_df], ignore_index=True)
df["General Flag"] = df["General Flag"].str.split(" :").str[0]
df.columns = df.columns.str.replace(" ", "_")

print(df)
new_output_dict = {}

for row in df.itertuples(index=False, name=None):
    row_dict = dict(zip(df.columns, row))

    new_key = str(row_dict["Subject"]) + str(row_dict["Timepoint"]) + str(row_dict["General_Flag"])
    new_output_dict.setdefault(new_key, {})

    for col, col_val in row_dict.items():
        new_output_dict[new_key].setdefault(col, col_val)
        if col == "Specific_Flags":
            if col in new_output_dict[new_key]:
                curr_val = new_output_dict[new_key][col]
                new_val = col_val
                new_val_list = str(new_val).split('|')
                curr_val_list = str(curr_val).split('|')
                new_val_list = [val for val in new_val_list if val not in curr_val_list]
                new_val_list.extend(curr_val_list)
                unique_vals = '|'.join(new_val_list)
                new_output_dict[new_key][col] = unique_vals
        elif col in ['Earliest_seen','Latest_seen']:
            date_format = "%Y-%m-%d"
            new_date = datetime.strptime(\
            col_val.split(' ')[0], date_format)
            old_date = datetime.strptime(\
            new_output_dict[new_key][col].split(' ')[0],
            date_format)

            if ((col == 'Earliest_seen' and new_date < old_date) 
            or (col == 'Latest_seen' and new_date > old_date)):
                new_output_dict[new_key][col] = col_val

output_list = list(new_output_dict.values())
output_df = pd.DataFrame(output_list)
output_df.to_csv('pronet_fixed.csv',index = False)


pronet_df = pd.read_csv('pronet_fixed.csv', keep_default_na = False)
prescient_df = pd.read_csv('prescient_fixed.csv', keep_default_na = False)


pronet_forms = pronet_df['General_Flag'].tolist()
prescient_forms = prescient_df['General_Flag'].tolist()

all_forms = pronet_forms.copy()
all_forms.extend(prescient_forms)

all_forms = [form.lower().replace(' ','_') for form in all_forms]

all_forms =  [form.lstrip('_') for form in all_forms]
unique_forms = list(set(all_forms))
print('forms checked')
print(unique_forms)
print(len(unique_forms))

data_dict = pd.read_csv('/home/ob001/refactored_qc/dependencies/data_dictionary/current_data_dictionary.csv',keep_default_na = False)

data_dict_forms = list(set(data_dict['Form Name'].tolist()))
print('total form count')
print(len(data_dict_forms))

forms_not_checked = [form for form in data_dict_forms if form not in unique_forms]
print('forms not checked')
print(forms_not_checked)

print(len(forms_not_checked))