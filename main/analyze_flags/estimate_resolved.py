import pandas as pd
import os
import sys 
import json
import numpy as np
import dropbox
from io import BytesIO
from datetime import datetime, timedelta, timezone
from xml.etree.ElementTree import ParseError as XMLParseError

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
        self.output_path = self.config_info['paths']['output_path']
        if self.config_info["testing_enabled"] == "True":
            self.output_path += "testing/"
            self.dropbox_path = f'/Apps/Automated QC Trackers/refactoring_tests/'
        else:
            self.dropbox_path = f'/Apps/Automated QC Trackers/'
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
                # Get all available revisions (set days_back=None) or specify a number like days_back=730 for 2 years
                self.recover_old_flags(combined_output, days_back=None)

    def recover_old_flags(self, path, days_back=None, page_limit=100, sample_every_days=4):
        """
        Recover history over time, paging beyond 100 revisions via before_rev.
        Requires path-mode (mode='path').

        path: Dropbox path string like "/folder/file.xlsx" (NOT file id)
        days_back: Number of days to go back (None = get all available revisions)
        page_limit: Maximum revisions per API call
        sample_every_days: Only keep revisions spaced at least this many days apart
        """

        dbx = self.utils.collect_dropbox_credentials()
        # Ensure access token is set when using refresh token (required before request_json_object)
        if hasattr(dbx, "check_and_refresh_access_token"):
            dbx.check_and_refresh_access_token()

        # --- find an object that can make raw RPC calls (SDK 11+: Dropbox has request_json_object itself) ---
        def find_requester(client):
            if hasattr(client, "request_json_object"):
                return client
            # older SDKs: look for internal transport/client
            for attr in ("_transport", "_client", "_request", "_session", "_Dropbox__transport", "_Dropbox__client"):
                obj = getattr(client, attr, None)
                if obj is not None and hasattr(obj, "request_json_object"):
                    return obj
            for name in dir(client):
                if name.startswith("__"):
                    continue
                try:
                    obj = getattr(client, name)
                except Exception:
                    continue
                if obj is not None and hasattr(obj, "request_json_object"):
                    return obj
            return None

        requester = find_requester(dbx)
        if requester is None:
            raise RuntimeError(
                "Couldn't find an internal requester on the Dropbox client with request_json_object(). "
                "In this SDK, easiest fix is upgrading the `dropbox` package, OR pass an access token "
                "and use requests directly."
            )

        # auth_type constant differs across SDK versions; try a few
        USER_AUTH = getattr(dropbox.dropbox_client, "USER_AUTH", None)
        if USER_AUTH is None:
            USER_AUTH = getattr(dropbox.dropbox_client, "AuthType", None)
            USER_AUTH = getattr(USER_AUTH, "USER", None) if USER_AUTH else None
        if USER_AUTH is None:
            # last resort: some versions accept string "user"
            USER_AUTH = "user"

        # Set cutoff if days_back is specified, otherwise get all available revisions
        cutoff = None
        if days_back is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            print(f"Retrieving revisions from the last {days_back} days (cutoff: {cutoff})")
        else:
            print("Retrieving all available revisions (no time limit)")

        cols = [
            "Subject", "Timepoint", "Additional Comments",
            "Manually Marked as Resolved", "Date Resolved", "General Flag", "Specific Flags"
        ]

        before_rev = None
        has_more = True
        last_kept_when = None
        kept = 0
        oldest_revision = None
        newest_revision = None

        # Collect unique rows by cols; track Earliest seen and Latest seen per row
        unique_rows = {}  # key (tuple of col values) -> (earliest_when, latest_when)

        while has_more:
            req = {"path": path, "mode": "path", "limit": page_limit}
            if before_rev:
                req["before_rev"] = before_rev

            # Some SDKs require request_binary and auth_type; others ignore extras.
            data = requester.request_json_object(
                "api",
                "files/list_revisions",
                "rpc",
                req,
                USER_AUTH,
                None,  # request_binary
            )

            entries = data.get("entries", [])
            has_more = bool(data.get("has_more", False))
            if not entries:
                break

            for entry in entries:
                rev = entry["rev"]
                when = datetime.fromisoformat(entry["server_modified"].replace("Z", "+00:00"))

                # Track oldest and newest revisions found
                if oldest_revision is None or when < oldest_revision:
                    oldest_revision = when
                if newest_revision is None or when > newest_revision:
                    newest_revision = when

                # stop once we're older than cutoff (if cutoff is set)
                if cutoff is not None and when < cutoff:
                    has_more = False
                    print(f"Reached cutoff date ({cutoff}), stopping retrieval")
                    break

                # optional sampling to "skip days"
                if sample_every_days and last_kept_when is not None:
                    if (last_kept_when - when).days < sample_every_days:
                        continue

                try:
                    md2, resp = dbx.files_download(path=path, rev=rev)
                    df = pd.read_excel(BytesIO(resp.content), keep_default_na=False)

                    # Support both old and new combined-output schemas, normalize to new names
                    old_cols = [
                        "Participant", "Site Comments", "Network Comments",
                        "Manually Resolved", "Date Resolved", "Form", "Flags"
                    ]
                    new_cols = [
                        "Subject", "Timepoint", "Additional Comments",
                        "Manually Marked as Resolved", "Date Resolved", "General Flag", "Specific Flags"
                    ]

                    if all(c in df.columns for c in new_cols):
                        # Already in new schema
                        cols = new_cols
                    elif all(c in df.columns for c in old_cols):
                        # Old schema – rename into new schema so everything is consistent
                        df = df.rename(columns={
                            "Participant": "Subject",
                            "Site Comments": "Additional Comments",
                            # If you want to preserve Network Comments separately, you can map it to a new column;
                            # here we just keep Additional Comments as the main free-text field.
                            "Manually Resolved": "Manually Marked as Resolved",
                            "Form": "General Flag",
                            "Flags": "Specific Flags",
                        })
                        cols = new_cols
                    else:
                        print(f"WARNING: Skipping revision {rev} from {when} - unrecognized columns: {list(df.columns)}")
                        continue

                    tmp = df[cols].replace(r"^\\s*$", pd.NA, regex=True)
                    mask_any_nonblank = tmp.notna().any(axis=1)
                    df_filtered = df[mask_any_nonblank].copy()

                    # Keep only rows where Date Resolved and Manually Marked as Resolved are both blank
                    mask_both_blank = pd.Series(True, index=df_filtered.index)
                    for c in ("Date Resolved", "Manually Marked as Resolved"):
                        if c in df_filtered.columns:
                            mask_both_blank &= df_filtered[c].fillna("").astype(str).str.strip() == ""
                    df_filtered = df_filtered[mask_both_blank].reset_index(drop=True)

                    # Track unique rows by cols; Earliest = min(revision date), Latest = max(revision date)
                    df_subset = df_filtered[cols].copy()
                    for row in df_subset.itertuples(index=False, name=None):
                        key = row
                        if key not in unique_rows:
                            unique_rows[key] = (when, when)
                        else:
                            e, l = unique_rows[key]
                            unique_rows[key] = (min(e, when), max(l, when))

                    last_kept_when = when
                    kept += 1
                    print(f"{path} - Revision {kept}: {rev} from {when}")
                except (ValueError, XMLParseError, Exception) as e:
                    # Skip corrupted or unreadable file revisions
                    print(f"WARNING: Skipping revision {rev} from {when} - file error: {type(e).__name__}: {str(e)[:100]}")
                    continue

            # paginate older than oldest in this page
            before_rev = entries[-1]["rev"]

        # Final output: unique rows with Earliest seen and Latest seen
        rows_with_dates = [
            (*key, earliest, latest)
            for key, (earliest, latest) in unique_rows.items()
        ]
        self.master = pd.DataFrame(rows_with_dates, columns=cols + ["Earliest seen", "Latest seen"])
        print(f"Total unique rows collected: {len(self.master)}")
        self.master.to_csv("recovered_flags_new_pronet.csv", index=False)
        
        # Print summary
        if oldest_revision and newest_revision:
            days_span = (newest_revision - oldest_revision).days
            print(f"\nDone. Kept {kept} revisions.")
            print(f"Date range: {oldest_revision} to {newest_revision} ({days_span} days)")
            if cutoff:
                print(f"Requested: {days_back} days back from today")
        else:
            print(f"\nDone. Kept {kept} revisions.")
    
if __name__ == '__main__':
    ResolvedEstimator().run_script()