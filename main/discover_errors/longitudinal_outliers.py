import numpy as np
import pandas as pd


class LongitudinalOutlierDetector:
    """
    Detect longitudinal outliers in long-format data where each row is one
    subject at one timepoint.

    Main behavior:
    - reads data from csv_path provided in __init__
    - keeps only columns that can be coerced to numeric and whose max > 4
    - computes within-subject expected values using the subject median
    - computes residual = observed - expected
    - scores each variable using a robust MAD-based z-score
    - if MAD is 0 or missing, falls back to absolute residual ranking
    - returns only the top 10,000 most severe flags, ranked descending

    Expected data structure:
    - one column identifying subject
    - optional time column for sorting within subject
    - one row per subject per timepoint
    """

    def __init__(
        self,
        csv_path,
        subject_col,
        time_col=None,
        min_points_per_subject=2,
        z_threshold=3.0,
        max_flags=10000,
    ):
        self.csv_path = "/home/ob001/refactored_qc/dependencies/merged_data.csv"
        self.subject_col = subject_col
        self.time_col = time_col
        self.min_points_per_subject = min_points_per_subject
        self.z_threshold = z_threshold
        self.max_flags = max_flags

        self.df = pd.read_csv(self.csv_path)

        self.outlier_table = None
        self.subject_summary = None
        self.subject_variable_summary = None
        self.full_long_table = None
        self.variables_used = None

    @staticmethod
    def robust_mad(x):
        """
        Median absolute deviation scaled
        to be comparable to standard deviation.
        """
        x = pd.Series(x).dropna()
        if len(x) == 0:
            return np.nan
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return 1.4826 * mad

    def _prepare_data(self):
        """
        Sort data within subject by time if a time column is provided.
        """
        data = self.df.copy()

        if self.subject_col not in data.columns:
            raise ValueError(f"subject_col '{self.subject_col}' not found in CSV columns")

        if self.time_col is not None and self.time_col not in data.columns:
            raise ValueError(f"time_col '{self.time_col}' not found in CSV columns")

        if self.time_col is not None:
            data = data.sort_values([self.subject_col, self.time_col]).reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)

        data["_original_row"] = np.arange(len(data))
        return data

    def get_numeric_variables(self, variables=None):
        """
        Return columns that can be meaningfully coerced to numeric and have max > 4.
        """
        data = self.df.copy()

        excluded = {self.subject_col}
        if self.time_col is not None:
            excluded.add(self.time_col)

        if variables is None:
            candidate_cols = [c for c in data.columns if c not in excluded]
        else:
            candidate_cols = [c for c in variables if c in data.columns and c not in excluded]

        filtered = []
        for c in candidate_cols:
            numeric_series = pd.to_numeric(data[c], errors="coerce")

            # Require at least one numeric value
            if numeric_series.notna().sum() == 0:
                continue

            col_max = numeric_series.max(skipna=True)
            if pd.notna(col_max) and col_max > 4:
                filtered.append(c)

        self.variables_used = filtered
        return filtered

    def fit(self, variables=None):
        """
        Detect outliers and store results on the object.
        Only the top `self.max_flags` row-level flags are retained in outlier_table.
        """
        data = self._prepare_data()
        variables = self.get_numeric_variables(variables=variables)

        print("Variables being tested:", variables)

        results = []

        for var in variables:
            keep_cols = [self.subject_col, "_original_row"]
            if self.time_col is not None:
                keep_cols.append(self.time_col)
            keep_cols.append(var)

            temp = data[keep_cols].copy()
            temp[var] = pd.to_numeric(temp[var], errors="coerce")

            pieces = []
            for sid, g in temp.groupby(self.subject_col, sort=False):
                g = g.copy()
                vals = pd.to_numeric(g[var], errors="coerce")

                if vals.notna().sum() < self.min_points_per_subject:
                    g["expected"] = np.nan
                    g["residual"] = np.nan
                    pieces.append(g)
                    continue

                # More stable than neighbor interpolation for sparse or irregular data
                subject_median = vals.median()
                g["expected"] = subject_median
                g["residual"] = vals - subject_median
                pieces.append(g)

            temp2 = pd.concat(pieces, axis=0)

            center = np.nanmedian(temp2["residual"])
            scale = self.robust_mad(temp2["residual"])

            # Fallback so variables do not disappear when MAD is 0
            if pd.isna(scale) or scale == 0:
                temp2["robust_z"] = np.nan
                temp2["severity"] = temp2["residual"].abs()
                temp2["score_type"] = "abs_residual_fallback"
                temp2["is_outlier"] = temp2["severity"] > 0
            else:
                temp2["robust_z"] = (temp2["residual"] - center) / scale
                temp2["severity"] = temp2["robust_z"].abs()
                temp2["score_type"] = "robust_z"
                temp2["is_outlier"] = temp2["severity"] >= self.z_threshold

            temp2["variable"] = var
            temp2["observed_value"] = temp2[var]

            print(
                f"{var}: "
                f"non-missing residuals={temp2['residual'].notna().sum()}, "
                f"scale={scale}, "
                f"flags={int(temp2['is_outlier'].sum())}"
            )

            out_cols = [
                self.subject_col,
                "_original_row",
                "variable",
                "observed_value",
                "expected",
                "residual",
                "robust_z",
                "severity",
                "score_type",
                "is_outlier",
            ]
            if self.time_col is not None:
                out_cols.insert(2, self.time_col)

            results.append(temp2[out_cols])

        if len(results) == 0:
            self.full_long_table = pd.DataFrame()
            self.outlier_table = pd.DataFrame()
            self.subject_summary = pd.DataFrame()
            self.subject_variable_summary = pd.DataFrame()
            return self

        self.full_long_table = pd.concat(results, axis=0, ignore_index=True)

        self.outlier_table = (
            self.full_long_table.loc[self.full_long_table["is_outlier"]]
            .sort_values("severity", ascending=False)
            .head(self.max_flags)
            .reset_index(drop=True)
        )

        self.subject_summary = (
            self.full_long_table.groupby(self.subject_col, as_index=False)
            .agg(
                n_outliers=("is_outlier", "sum"),
                max_severity=("severity", "max"),
                median_severity=("severity", "median"),
            )
            .sort_values(["max_severity", "n_outliers"], ascending=[False, False])
            .reset_index(drop=True)
        )

        self.subject_variable_summary = (
            self.full_long_table.groupby([self.subject_col, "variable"], as_index=False)
            .agg(
                n_outliers=("is_outlier", "sum"),
                max_severity=("severity", "max"),
                median_severity=("severity", "median"),
            )
            .sort_values(["max_severity", "n_outliers"], ascending=[False, False])
            .reset_index(drop=True)
        )

        return self

    def get_ranked_outliers(self):
        """
        Return the top max_flags row-level
        outliers, ranked most to least severe.
        """
        if self.outlier_table is None:
            return pd.DataFrame()
        return self.outlier_table.copy()

    def get_subject_summary(self):
        """
        Return ranked subject-level summary.
        """
        if self.subject_summary is None:
            return pd.DataFrame()
        return self.subject_summary.copy()

    def get_subject_variable_summary(self):
        """
        Return ranked subject-variable summary.
        """
        if self.subject_variable_summary is None:
            return pd.DataFrame()
        return self.subject_variable_summary.copy()

    def get_all_scores(self):
        """
        Return all evaluated observations with scores.
        """
        if self.full_long_table is None:
            return pd.DataFrame()
        return self.full_long_table.copy()

    def save_results(
        self,
        outlier_path="ranked_longitudinal_outliers_top_10000.csv",
        subject_summary_path="ranked_subject_outlier_summary.csv",
        subject_variable_summary_path="ranked_subject_variable_outliers.csv",
        all_scores_path="all_longitudinal_outlier_scores.csv",
    ):
        """
        Save result tables to CSV.
        """
        if self.outlier_table is not None:
            self.outlier_table.to_csv(outlier_path, index=False)
        #if self.subject_summary is not None:
        #    self.subject_summary.to_csv(subject_summary_path, index=False)
        #if self.subject_variable_summary is not None:
        #    self.subject_variable_summary.to_csv(subject_variable_summary_path, index=False)
        #if self.full_long_table is not None:
        #    self.full_long_table.to_csv(all_scores_path, index=False)


# =========================
# Example usage
# =========================

detector = LongitudinalOutlierDetector(
    csv_path="your_data.csv",
    subject_col="subjectid",
    time_col="event_name",   # set to None if rows are already ordered correctly
    min_points_per_subject=2,
    z_threshold=3.0,
    max_flags=10000,
)

detector.fit()

ranked_outliers = detector.get_ranked_outliers()
subject_summary = detector.get_subject_summary()
subject_variable_summary = detector.get_subject_variable_summary()
all_scores = detector.get_all_scores()

print("\nVariables used:")
print(detector.variables_used)

print("\nTop flagged observations (capped at 10,000, ordered most to least severe):")
print(ranked_outliers.head(50))

print("\nSubjects ranked by strongest outlier:")
print(subject_summary.head(25))

print("\nSubject-variable combinations ranked by strongest outlier:")
print(subject_variable_summary.head(25))

detector.save_results()