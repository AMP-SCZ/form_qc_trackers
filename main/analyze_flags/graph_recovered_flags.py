import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os
import json
import sys
import statistics as stats
import re

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
print(parent_dir)
from utils.utils import Utils

class ResolvedGrapher():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.output_path = self.config_info['paths']['output_path']
        self.total_flags = {'ProNET':0,'PRESCIENT':0}
        self.large_jumps = {}
        self.excluded_dates = {'ProNET':{'orig':[],'new':[]},
        'PRESCIENT':{'orig':[],'new':[]}}

    def run_script(self):
        for file in ['prescient_fixed.csv', 'pronet_fixed.csv']:
            if 'pronet' in file:
                network = 'ProNET'
            else:
                network = 'PRESCIENT'
            if 'orig' in file:
                version = 'orig'
            else:
                version = 'new'

            self.resolved_df = pd.read_csv(file, keep_default_na = False)
            self.resolved_df.columns = self.resolved_df.columns.str.replace(" ", "_")

            self.resolved_df['Latest_seen'] = pd.to_datetime(
            self.resolved_df['Latest_seen'], errors="coerce")

            self.resolved_df = self.resolved_df.sort_values(
            by='Latest_seen', ascending=True)

            strings_to_match = ["p1p8", "p9ac32", "mood"]
            col = "General_Flag"

            pattern = "|".join(re.escape(s) for s in strings_to_match)

            self.resolved_df = self.resolved_df[self.resolved_df[
            col].astype(str).str.contains(pattern, case=False, na=False)]
            self.resolved_df.to_csv(
            self.output_path + file.replace('.csv','') + 'filtered.csv',
            index = False)

            self.flags_per_day = {}
            self.avrg_days_btwn = {}

            self.collect_errors_per_day_new(self.resolved_df, network,version)
        
        print(self.large_jumps)
        print(self.total_flags)

    def collect_errors_per_day_new(self, df, network, version):
        # Build per-date buckets first (order-independent processing)
        per_date = {}

        for row in df.itertuples():
            date = getattr(row, 'Latest_seen')
            earliest_date = getattr(row, 'Earliest_seen')

            days_btwn = self.utils.find_days_between(str(date), str(earliest_date))

            # Normalize flag count safely
            flag_str = str(getattr(row, 'Specific_Flags', '') or '')
            flag_count = flag_str.count(':') 
            #len([f for f in flag_str.split('|') if f.strip()])

            per_date.setdefault(date, []).append({
                'days_btwn': days_btwn,
                'flag_count': flag_count
            })

        self.flags_per_day = {}
        self.avrg_days_btwn = {}

        # Process each day deterministically
        for date in sorted(per_date.keys()):
            rows = per_date[date]

            day_vals = [r['days_btwn'] for r in rows if isinstance(r['days_btwn'], (int, float))]
            if not day_vals:
                continue

            # Full-day stats (stable, not running)
            day_count = len(day_vals)
            day_mean = sum(day_vals) / day_count
            day_std = stats.stdev(day_vals) if day_count > 1 else 0

            # Apply your existing "large jump" style rules using full-day stats
            # (interpretation of your current thresholds)
            is_large_jump = (
                ((day_count > 100 and day_mean < 7) or (day_count >= 300)) or
                (day_count > 50 and float(day_std) < 4)
            )

            if is_large_jump:
                self.large_jumps.setdefault(network, {})
                self.large_jumps[network].setdefault(version, [])
                if date not in self.large_jumps[network][version]:
                    self.large_jumps[network][version].append(date)
                continue

            # Otherwise keep rows (optionally cap at 300 to preserve your cap)
            kept_rows = rows[:300]

            self.flags_per_day[date] = len(kept_rows)

            vals = [r['days_btwn'] for r in kept_rows if isinstance(r['days_btwn'], (int, float))]
            total = sum(vals)
            count = len(vals)
            avrg = (total / count) if count else 0
            stdv = stats.stdev(vals) if len(vals) > 1 else 0

            self.avrg_days_btwn[date] = {
                'total': total,
                'count': count,
                'avrg': avrg,
                'all_vals': vals,
                'stdv': stdv
            }

            # Update total flags only for kept rows
            for r in kept_rows:
                self.total_flags[network] += r['flag_count']

        x = list(self.flags_per_day.keys())
        y = list(self.flags_per_day.values())

        plt.plot(x, y)
        plt.savefig(f"resolved_over_time_{network}_{version}.png",
        format="png", bbox_inches="tight")
        plt.close()

    def collect_errors_per_day(self,df, network, version):
        for row in df.itertuples():
            date = getattr(row,'Latest_seen')
            if str(date) in self.excluded_dates[network][version]:
                print('skip')
                continue
            earliest_date = getattr(row,'Earliest_seen')
            days_btwn = self.utils.find_days_between(str(date),str(earliest_date))

            flag_str = getattr(row,'Specific_Flags')
            flag_count = flag_str.count('|') + 1
                
            self.flags_per_day.setdefault(date, 0)
            self.avrg_days_btwn.setdefault(date, {'total':0,'count': 0, 'avrg':0,'all_vals':[],'stdv':0})
            if ((not (self.flags_per_day[date] > 100 and 
            self.avrg_days_btwn[date]['avrg'] < 7)) and (self.flags_per_day[date] < 300)):
                if (not (self.avrg_days_btwn[date]['count'] > 50
                and float(self.avrg_days_btwn[date]['stdv']) < 4)):
                    self.flags_per_day[date]+=1 
                    self.avrg_days_btwn[date]['total']+= days_btwn
                    self.avrg_days_btwn[date]['all_vals'].append(days_btwn)
                    if len(self.avrg_days_btwn[date]['all_vals']) > 2:
                        self.avrg_days_btwn[date]['stdv'] = stats.stdev(self.avrg_days_btwn[date]['all_vals'])      

                    self.avrg_days_btwn[date]['count'] +=1 
                    self.avrg_days_btwn[date]['avrg'] =  (
                    self.avrg_days_btwn[date]['total']
                    / self.avrg_days_btwn[date]['count'])
                    self.total_flags[network] += flag_count
                    
                    """if self.flags_per_day[date] > 200:
                        print(network)
                        print(version)
                        print(date)
                        print('----------------')"""
                else:
                    self.excluded_dates[network][version].append(str(date))
                
            else:
                self.excluded_dates[network][version].append(str(date))
                self.large_jumps.setdefault(network, {})
                self.large_jumps[network].setdefault(version, [])
                if date not in self.large_jumps[network][version]:
                    self.large_jumps[network][version].append(date)
        x = list(self.flags_per_day.keys())
        y = list(self.flags_per_day.values())

        plt.plot(x, y)

        plt.savefig(f"resolved_over_time_{network}_{version}.png", format="png", bbox_inches="tight")
        # plt.show()  # optional
        plt.close()


if __name__ == '__main__':
    ResolvedGrapher().run_script()