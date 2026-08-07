import pandas as pd 
import json 

with open(
"/home/ob001/refactored_qc/form_qc_trackers/_audit_tmp/symptom_variable_map_rows.json",
"r") as f:
    symptom_maps_json = json.load(f)


refined_mapping = []

for var_data in symptom_maps_json['matches']:
    var = var_data["variable"]
    label = var_data["field_label"]
    form = var_data["form"]
    if form in ["coenrollment_form"]:
        continue
    if "chrpharm" in var_data["variable"]:
        continue
    if  var_data["field_type"] in ["descriptive","notes"]:
        continue 
    else:
        refined_mapping.append({"variable":var,"label":label, 
        "form":var_data["form"],
        "clinical_domain":var_data["domain"]})

refined_mapping_df = pd.DataFrame(refined_mapping)
refined_mapping_df.to_csv('clinical_var_mappings.csv', index = False)