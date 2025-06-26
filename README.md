# AMP SCZ Form Quality Control Trackers

This repository contains a quality control (QC) framework developed for the **Accelerating Medicines Partnership for Schizophrenia (AMP SCZ)** project—a large-scale, longitudinal study on schizophrenia. 

##  Overview

The QC system:
- Analyzes datasets for missing, unexpected, or inconsistent values
- Detects mismatches between variable values and form specifications
- Flags potential outliers and tracks resolved errors
- Generates detailed reports for quality tracking over time

##  Project Structure

```
form_qc_trackers/
├── run_qc.py                     # Main entry point
├── config.json                   # Customizable parameters for each run
├── logging_config.py             # Logging setup
├── main/
│   ├── analyze_dataset/          # Analyzes structure of data
│   ├── discover_errors/          # Searches for anomalies and potential errors
│   └── generate_reports/         # Creates and uploads output to Dropbox
├── README.md
└── LICENSE
```

##  Getting Started


3. **Configure your run**
   - Modify `config.json` to define paths.
   - download packages specified in requirements.txt
   - Include dependencies listed below
   in dependencies folder that you define in
   the config file

4. **Run the pipeline**
   
   - Files Needed (all other dependencies will be generated)
      dependencies./

      ├── dropbox_credentials.json                    
      ├── scid_diagnosis_vars.json  
      ├── forms_per_timepoint.json                
      ├── data_dictionary/                                       
      │   └── current_data_dictionary.csv 
   - Execute the "run_qc.py" file