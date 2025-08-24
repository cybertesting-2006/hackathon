"""
report.py (Gradio version)
Generates report objects that can be displayed in a Gradio UI
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ReportGenerator:
    def _init_(self, cleaned_df: pd.DataFrame, raw_df: pd.DataFrame = None, report_dir="reports"):
        self.df = cleaned_df
        self.raw_df = raw_df
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        (self.report_dir / "plots").mkdir(exist_ok=True)
        self.summary = {}

    # Dataset Overview
    def dataset_overview(self):
        self.summary["shape"] = self.df.shape
        self.summary["columns"] = list(self.df.columns)
        self.summary["dtypes"] = self.df.dtypes.astype(str).to_dict()
        self.summary["missing_values"] = self.df.isna().sum().to_dict()
        self.summary["duplicates"] = self.df.duplicated().sum()
        return pd.DataFrame.from_dict(self.summary["missing_values"], orient="index", columns=["Missing Values"])

    # Compare Before vs After
    def compare_before_after(self):
        if self.raw_df is None:
            return None
        comparison = {
            "Rows Before": self.raw_df.shape[0],
            "Rows After": self.df.shape[0],
            "Columns Before": self.raw_df.shape[1],
            "Columns After": self.df.shape[1],
            "Missing Before": int(self.raw_df.isna().sum().sum()),
            "Missing After": int(self.df.isna().sum().sum()),
            "Duplicates Before": int(self.raw_df.duplicated().sum()),
            "Duplicates After": int(self.df.duplicated().sum())
        }
        return pd.DataFrame(list(comparison.items()), columns=["Metric", "Value"])

    # Basic Statistics
    def basic_statistics(self):
        return self.df.describe(include="all", datetime_is_numeric=True).transpose()

    # Outlier Summary
    def outlier_summary(self):
        outliers = {}
        for col in self.df.select_dtypes(include="number").columns:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers[col] = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
        return pd.DataFrame.from_dict(outliers, orient="index", columns=["Outliers"])

    # Correlation Heatmap
    def correlation_heatmap(self):
        plt.figure(figsize=(7, 5))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plot_path = self.report_dir / "plots/correlation_heatmap.png"
        plt.savefig(plot_path)
        plt.close()
        return str(plot_path)

    # Column Distributions
    def column_distributions(self):
        paths = []
        for col in self.df.select_dtypes(include="number").columns:
            plt.figure()
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plot_path = self.report_dir / f"plots/{col}_hist.png"
            plt.savefig(plot_path)
            plt.close()
            paths.append(str(plot_path))
        return paths

    # Generate Everything
    def generate_report(self):
        overview = self.dataset_overview()
        comparison = self.compare_before_after()
        stats = self.basic_statistics()
        outliers = self.outlier_summary()
        heatmap = self.correlation_heatmap()
        distributions = self.column_distributions()

        return {
            "overview": overview,
            "comparison": comparison,
            "stats": stats,
            "outliers": outliers,
            "heatmap": heatmap,
            "distributions": distributions
        }
