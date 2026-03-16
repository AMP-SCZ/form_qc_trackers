import os
import sys
import json
import csv
import traceback
from pathlib import Path
from collections import Counter

import pandas as pd
from frictionless import validate, Schema, system

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils


class LLMQcFlagFinder:
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path

        with open(f"{self.absolute_path}/config.json", "r") as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info["paths"]["combined_csv_path"]
        self.output_path = self.config_info["paths"]["output_path"]
        if self.config_info.get("testing_enabled") == "True":
            self.output_path += "testing/"

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Where cleaned copies will go (originals untouched)
        self.cleaned_dir = Path(self.output_path) / "cleaned_csvs"
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        # Minimal starter schema (expand later)
        self.schema = Schema.from_descriptor(
            {"fields": [{"name": "subjectid", "type": "string", "constraints": {"required": True}}]}
        )
        self.schema = None
        # Combined summary rows
        self.qc_rows: list[dict] = []

        # encodings to try when reading "problem" CSVs
        self.encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    def run_script(self):
        self.loop_csv_files()
        self.save_combined_qc_summary()

    # -----------------------------
    # Frictionless helpers
    # -----------------------------
    @staticmethod
    def _iter_report_errors(report):
        tasks = getattr(report, "tasks", None)
        if tasks:
            for task in tasks:
                for err in getattr(task, "errors", []) or []:
                    yield err
        for err in (getattr(report, "errors", []) or []):
            yield err

    def frictionless_validate(self, path: str, schema: Schema | None):
        """Validate with trusted=True so absolute paths are allowed."""
        with system.use_context(trusted=True):
            return validate(path, schema=schema)

    def summarize_report(self, report, max_preview=5):
        """Return (valid, error_count, error_types_summary, first_errors_str)."""
        types = []
        msgs = []
        for e in self._iter_report_errors(report):
            t = getattr(e, "type", "unknown")
            m = getattr(e, "message", "")
            types.append(t)
            if len(msgs) < max_preview:
                msgs.append(f"{t}: {m}")

        type_counts = Counter(types)
        type_summary = "; ".join([f"{t}={c}" for t, c in type_counts.most_common()]) if type_counts else ""
        return bool(report.valid), int(sum(type_counts.values())), type_summary, " | ".join(msgs)

    # -----------------------------
    # Cleaning / fixing CSVs
    # -----------------------------
    def _make_unique_headers(self, headers):
        """
        - remove blank headers
        - make duplicates unique by suffixing __2, __3, ...
        """
        cleaned = []
        counts = {}

        for h in headers:
            h = (h or "").strip()
            if h == "":
                # drop empty labels entirely
                continue

            if h not in counts:
                counts[h] = 1
                cleaned.append(h)
            else:
                counts[h] += 1
                cleaned.append(f"{h}__{counts[h]}")

        return cleaned

    def _read_first_row_with_encoding(self, csv_path: Path):
        """
        Try encodings until we can read the first row (header).
        Returns (encoding_used, header_list, dialect)
        """
        last_err = None
        for enc in self.encodings_to_try:
            try:
                with open(csv_path, "r", encoding=enc, newline="") as f:
                    sample = f.read(8192)
                    f.seek(0)
                    dialect = csv.Sniffer().sniff(sample)
                    reader = csv.reader(f, dialect)
                    header = next(reader)
                return enc, header, dialect
            except Exception as e:
                last_err = e
                continue
        raise last_err

    def clean_csv_to_new_file(self, csv_path: str) -> dict:
        """
        Writes a cleaned version to self.cleaned_dir.
        Fixes:
          - encoding by decoding with a working encoding and re-writing as UTF-8
          - extra blank header labels
          - ragged rows (extra-cell / missing cells)
        Returns metadata about the cleaning.
        """
        src = Path(csv_path)
        if not src.exists():
            return {"cleaned_path": "", "used_encoding": "", "fixed": False, "fix_notes": "missing-file"}

        used_enc, raw_header, dialect = self._read_first_row_with_encoding(src)
        new_header = self._make_unique_headers(raw_header)

        # If we dropped blank headers, the expected column count changes.
        # We need a mapping from old index -> new index for the kept headers.
        kept_indices = [i for i, h in enumerate(raw_header) if (h or "").strip() != ""]
        target_n = len(new_header)

        dst = self.cleaned_dir / src.name  # same filename, different folder
        fix_notes = []

        if len(new_header) != len(raw_header):
            fix_notes.append(f"dropped_blank_headers:{len(raw_header) - len(new_header)}")

        # Now rewrite the entire file in UTF-8 with corrected header + aligned rows
        with open(src, "r", encoding=used_enc, newline="") as fin, open(dst, "w", encoding="utf-8", newline="") as fout:
            reader = csv.reader(fin, dialect)
            writer = csv.writer(fout)

            # skip original header and write cleaned header
            _ = next(reader, None)
            writer.writerow(new_header)

            row_idx = 1
            for row in reader:
                row_idx += 1

                # Keep only cells corresponding to kept header indices
                filtered = [row[i] if i < len(row) else "" for i in kept_indices]

                # Truncate or pad to match header length
                if len(filtered) > target_n:
                    filtered = filtered[:target_n]
                    fix_notes.append("truncated_rows")
                elif len(filtered) < target_n:
                    filtered.extend([""] * (target_n - len(filtered)))
                    fix_notes.append("padded_rows")

                writer.writerow(filtered)

        return {
            "cleaned_path": str(dst),
            "used_encoding": used_enc,
            "fixed": True,
            "fix_notes": ",".join(sorted(set(fix_notes))) if fix_notes else "",
        }

    # -----------------------------
    # Main loop
    # -----------------------------
    def loop_csv_files(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(["floating", "conversion"])

        for network in ["PRESCIENT"]:
            for tp in tp_list:
                csv_path = (
                    f"{self.comb_csv_path}AMPSCZ-combined-redcap_"
                    f"{tp.replace('month','month_').replace('floating','floating_forms')}_{network}-day1to1.csv"
                )

                row_out = {
                    "file": csv_path,
                    "exists": Path(csv_path).exists(),
                    "cleaned_file": "",
                    "cleaned": False,
                    "used_encoding": "",
                    "fix_notes": "",
                    "valid_original": False,
                    "errors_original": 0,
                    "types_original": "",
                    "valid_cleaned": False,
                    "errors_cleaned": 0,
                    "types_cleaned": "",
                    "first_errors_cleaned": "",
                }

                try:
                    # Validate original (optional but useful for comparison)
                    if row_out["exists"]:
                        orig_report = self.frictionless_validate(csv_path, schema=self.schema)
                        v0, e0, t0, _ = self.summarize_report(orig_report, max_preview=3)
                        row_out["valid_original"] = v0
                        row_out["errors_original"] = e0
                        row_out["types_original"] = t0

                    # Clean to new file (does not touch original)
                    clean_meta = self.clean_csv_to_new_file(csv_path)
                    row_out.update({
                        "cleaned_file": clean_meta["cleaned_path"],
                        "cleaned": clean_meta["fixed"],
                        "used_encoding": clean_meta["used_encoding"],
                        "fix_notes": clean_meta["fix_notes"],
                    })

                    # Validate cleaned
                    if row_out["cleaned_file"]:
                        clean_report = self.frictionless_validate(row_out["cleaned_file"], schema=self.schema)
                        v1, e1, t1, fe1 = self.summarize_report(clean_report, max_preview=5)
                        row_out["valid_cleaned"] = v1
                        row_out["errors_cleaned"] = e1
                        row_out["types_cleaned"] = t1
                        row_out["first_errors_cleaned"] = fe1

                    self.qc_rows.append(row_out)

                    # Console status
                    if row_out["valid_cleaned"]:
                        print(f"[QC OK]   cleaned={row_out['cleaned_file']}")
                    else:
                        print(
                            f"[QC FAIL] {csv_path}  "
                            f"orig=({row_out['errors_original']}:{row_out['types_original']})  "
                            f"cleaned=({row_out['errors_cleaned']}:{row_out['types_cleaned']})"
                        )

                except Exception as e:
                    print("\n========================")
                    print("QC PIPELINE CRASHED")
                    print(f"File: {csv_path}")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    print("========================\n")

                    row_out["types_cleaned"] = "qc-crash=1"
                    row_out["first_errors_cleaned"] = str(e)
                    self.qc_rows.append(row_out)

    def save_combined_qc_summary(self):
        out_path = Path(self.output_path) / "frictionless_qc_summary.csv"
        df = pd.DataFrame(self.qc_rows)

        if not df.empty:
            # put most important columns first
            cols = [
                "file", "exists",
                "cleaned_file", "cleaned", "used_encoding", "fix_notes",
                "valid_original", "errors_original", "types_original",
                "valid_cleaned", "errors_cleaned", "types_cleaned",
                "first_errors_cleaned",
            ]
            df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
            df = df.sort_values(by=["valid_cleaned", "errors_cleaned", "file"], ascending=[True, False, True])

        df.to_csv(out_path, index=False)
        print(f"\n[SAVED] Combined QC summary: {out_path}")


if __name__ == "__main__":
    LLMQcFlagFinder().run_script()