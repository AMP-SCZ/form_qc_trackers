import pandas as pd
import matplotlib.pyplot as plt
import dropbox
from utils import Utils
from datetime import datetime
import json

class GraphErrors():
    """
    Class to calculate and 
    visualize the number of unresolved 
    errors from each site for both networks.
    """

    def __init__(self):
        self.tracker_abs_path = \
        '/PHShome/ob001/anaconda3/new_forms_qc/QC/site_outputs'

        self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
        self.utils = Utils()
        self.site_errors = {}

    def run_script(self) -> None:
        """
        Loops through both networks
        and calls functions to visualize
        the number of errors per site.
        """

        for network in ['PRONET','PRESCIENT']:
            tracker_path = \
            f'{self.tracker_abs_path}/{network}/combined/{network}_Output.xlsx'
            tracker_df = pd.read_excel(tracker_path, keep_default_na=False)
            self.collect_errors_per_site(tracker_df,network)
            self.visualize_errors(network)

    def collect_errors_per_site(
        self, df : pd.DataFrame, network : str
    ) -> None:
        
        """
        Collects number of unresolved errors
        from each site and categorizes them
        by errors that were detected less than
        or more than 30 days ago.
        
        Parameters
        --------------
        df : pd.DataFrame
            Dataframe with Errors
        network : str
            Current network (PRONET or PRESCIENT) 
        """

        df.columns = df.columns.str.replace(' ','_')
        for row in df.itertuples():
            self.site_errors.setdefault(network,{})
            self.site_errors[network].setdefault(\
            row.Subject[:2],{'under_thirty':0,'over_thirty':0})

            if getattr(row,'Date_Resolved') != '' or getattr(row,'Manually_Marked_as_Resolved') != '':
                continue
            
            date_detected = str(getattr(row,'Date_Flag_Detected')).split(' ')[0]
            curr_date = str(datetime.today()).split(' ')[0]
            diff = self.utils.calculate_days_between_dates(date_detected,curr_date)
            if diff > 30:
                self.site_errors[network][row.Subject[:2]]['over_thirty'] +=1
            else:
                self.site_errors[network][row.Subject[:2]]['under_thirty'] +=1


    def visualize_errors(self, network : str) -> None:
        """
        Creates stacked bar graph of 
        errors for each site of the
        current network. Saves graph
        as a PNG.

        Parameters
        ----------
        network : str
            Current network (PRONET or PRESCIENT)
        """

        categories = list(self.site_errors[network].keys())
        over_thirty = []
        under_thirty = []
        for site,errors in self.site_errors[network].items():
            over_thirty.append(errors['over_thirty'])
            under_thirty.append(errors['under_thirty'])

        plt.figure(figsize=(15, 9))  

        plt.bar(categories, over_thirty,
        label='Detected Over 30 Days Ago', color = '#DE3C20')

        plt.bar(categories, under_thirty, bottom=over_thirty,
        label='Detected Under 30 Days Ago', color = '#EA9A4A')
        plt.legend()
        plt.title((F'{network.replace("PRONET","ProNET")} Queries'
        ' Per Site (Any Forms That Have Unresolved'
        ' Queries Will Not Be Uploaded to the NDA)'))
        plt.xlabel('Site')
        plt.ylabel('Unresolved Queries')
        graph_path = f'{self.tracker_abs_path}/{network}/combined/{network}_queries_graph.png'
        plt.savefig(graph_path,dpi = 600)
        self.upload_to_dropbox(graph_path,network)

    def upload_to_dropbox(self, local_path : str, network : str) -> None:
        """
        Reads the graph from the local file
        and uploads it to dropbox.

        Parameters
        -----------------
        local_path : str
            Path to local graph file
        network : str
            current network (PRONET or PRESCIENT)
        """
        
        dbx = self.utils.collect_dropbox_credentials()
        with open(local_path, 'rb') as f:
            dbx.files_upload(f.read(),\
            f'/Apps/Automated QC Trackers/{network}/combined/{network}_queries_graph.png',\
            mode=dropbox.files.WriteMode.overwrite)


if __name__ == '__main__':
    GraphErrors().run_script()
