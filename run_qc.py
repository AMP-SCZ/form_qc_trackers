
import os
import pandas as pd


def define_paths(location = 'pnl_server'):
    """Defines paths depending on
    where the program is being run
    
    Parameters
    ------------
    location: where the program
    is executing

    Returns
    ------------
    combined_df_folder: path to combined CSVs
    absolute_path: absolute path of current program
    running
    """

    if location =='pnl_server':
        combined_df_folder = '/data/predict1/data_from_nda/formqc/'
        absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
    else:
        combined_df_folder = 'C:/formqc/AMPSCZ_QC_and_Visualization/QC/combined_csvs/'
        absolute_path = ''

    return combined_df_folder,absolute_path

def initalize_output_per_timepoint():
    """Organizes every timepoint as a key
    in a dictionary

    Returns
    ------------
    output_per_timepoint: dictionary
    with every timepoint as a key
    """

    output_per_timepoint = {'screen':[],'baseln':[]}
    for x in range(1,24):
      output_per_timepoint['month'+f'{x}'] = []
    output_per_timepoint['month18'] = []
    output_per_timepoint['month24'] = []

    return output_per_timepoint

def reformat_columns(output_per_timepoint,timepoint,report):
    """Reformats columns that consist of lists
    into strings separated by a pipe

    Parameters
    ---------
    output_per_timepoint: dictionary of QC output 
    timepoint: current timepoint in loop
    report: current report being reformatted

    Returns
    ---------
    output_per_timepoint: modified output
    """

    for row_ind in range(0,len(\
        output_per_timepoint[timepoint][report])):
        for col in ['Specific Flags','Variable Translations']:
            output_per_timepoint[timepoint][report][row_ind][col]\
            = "| ".join(output_per_timepoint[\
            timepoint][report][row_ind][col])

    return output_per_timepoint

def loop_timepoints():
    """Calls QC check on every timepoint
    for each network and organizes it into a final 
    output for the trackers"""

    combined_df_folder,absolute_path = define_paths()
    for network in ['PRONET','PRESCIENT']:
        final_output = {}
        output_per_timepoint =initalize_output_per_timepoint()
        for timepoint in output_per_timepoint.keys():
            print(timepoint)
            timepoint_str =\
            timepoint.replace('baseln','baseline').replace('screen','screening')
            filepath = (f'{combined_df_folder}'
            f'combined-{network}-{timepoint_str}-day1to1.csv')   
            if os.path.exists(filepath):
                formqc_check = IterateForms(filepath,f'{timepoint}','Main Report')
                output_per_timepoint[timepoint] = formqc_check.run_script()
                for report in output_per_timepoint[timepoint].keys():
                    output_per_timepoint =\
                    reformat_columns(output_per_timepoint,timepoint,report)
                    final_output.setdefault(report,[])
        for timepoint in output_per_timepoint.keys():
            if output_per_timepoint[timepoint] !=[]:
                for report in output_per_timepoint[timepoint].keys():
                    final_output[report].extend(\
                    output_per_timepoint[timepoint][report])

        create_trackers(final_output,network)

def create_trackers(final_output,network):
    """Calls GenerateTrackers class
    to format the final excel spreadsheets

    Parameters
    ------------
    final_output: list of dictionaries to be 
    
    converted into pandas dataframe
    """

    report_list = ['Main Report','Incomplete Forms','Secondary Report',\
    'Twenty One Day Tracker','Substantial Data Missing','Minor Data Missing','Blood Report',\
    'Cognition Report','Scid Report']

    print(final_output.keys())

    for report in report_list:
        if report in final_output.keys():
            print(report)
            data = pd.DataFrame(final_output[report])
            GenerateTrackers(data,network,report).run_script()


if __name__ == '__main__':
    loop_timepoints()
