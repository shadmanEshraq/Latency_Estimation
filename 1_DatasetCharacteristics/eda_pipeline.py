# eda_pipeline.py
"""
A comprehensive EDA pipeline for pandas DataFrames,
including functions for overview, summary statistics, missing value analysis,
outlier detection, skewness reporting, correlation heatmaps, distribution plots, boxplots, and target relationships.

ToDO: 1. you can add a memory optimization function later if you want to further enhance the pipeline.
ToDO: 2. Train/ Test drift analysis can be added as well if you want to compare two datasets (e.g. train vs test) and identify any differences in distributions or feature characteristics.

EDAPipeline
│
├── overview()
├── memory_report()
├── summary_stats()
├── missing_report()
├── cardinality_report()
├── detect_outliers_iqr()
├── skewness_report()
├── log_transform_candidates()
├── correlation_heatmap()
├── distribution_plots()
├── boxplots()
├── target_relationships()
├── optimize_memory()
└── run_all()

Dependency: pandas, numpy, matplotlib, seaborn, scipy

Example usage:

    from eda_pipeline import EDAPipeline
    eda = EDAPipeline(df = df, target='target_column')
    eda.run_all()


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme(style="whitegrid", context="talk", font_scale=1)
# sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
# sns.set_theme(style="ticks", context="notebook", font_scale=1.2)
# sns.set_theme(style="white", context="poster", font_scale=1)


from scipy.stats import skew

# Import plotting style (done on the notebook level to avoid issues with imports in the class)
# import plot_config


class EDAPipeline:
    def __init__(self, df, target=None):

        self.df = df.copy()
        self.target = target

        # Numerical
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        # Categorical
        self.categorical_cols = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Datetime
        self.datetime_cols = self.df.select_dtypes(
            include=["datetime"]
        ).columns.tolist()

    # =========================================================
    # BASIC OVERVIEW
    # =========================================================

    def overview(self):
        """Prints a basic overview of the dataset, including shape, columns, data types, missing values, and duplicates."""

        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print(f"Shape: {self.df.shape}")

        print("\nColumns:")
        print(self.df.columns.tolist())

        print("\nData Types:")
        print(self.df.dtypes)

        print("\nMissing Values:")
        print(self.df.isnull().sum().sort_values(ascending=False))

        print("\nDuplicate Rows:")
        print(self.df.duplicated().sum())

    # =========================================================
    # MEMORY USAGE
    # =========================================================

    def memory_report(self):
        """Prints the memory usage of the dataset in megabytes."""

        mem = self.df.memory_usage(deep=True).sum() / 1024**2

        print("=" * 60)
        print("MEMORY USAGE")
        print("=" * 60)

        print(f"Dataset memory usage: {mem:.2f} MB")

    # =========================================================
    # DESCRIPTIVE STATISTICS
    # =========================================================

    def summary_stats(self):
        """Prints summary statistics for numerical columns."""

        print("=" * 60)
        print("NUMERICAL SUMMARY")
        print("=" * 60)

        print(self.df[self.numeric_cols].describe().T)

    # =========================================================
    # MISSING VALUE REPORT
    # =========================================================

    def missing_report(self):
        """Prints a report of missing values for each column."""

        missing = pd.DataFrame(
            {
                "missing_count": self.df.isnull().sum(),
                "missing_percent": self.df.isnull().mean() * 100,
            }
        )

        missing = missing.sort_values(by="missing_percent", ascending=False)

        return missing

    # =========================================================
    # HIGH CARDINALITY REPORT
    # =========================================================

    def cardinality_report(self):
        """Prints a report of unique values and cardinality percentage for each categorical column."""

        report = {}

        for col in self.categorical_cols:
            report[col] = {
                "unique_values": self.df[col].nunique(),
                "cardinality_percent": self.df[col].nunique() / len(self.df) * 100,
            }

        return pd.DataFrame(report).T.sort_values(by="unique_values", ascending=False)

    # =========================================================
    # OUTLIER DETECTION
    # =========================================================

    def detect_outliers_iqr(self):
        """Detects outliers in numerical columns using the IQR method."""

        outlier_report = {}

        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]

            outlier_report[col] = len(outliers)

        return pd.DataFrame.from_dict(
            outlier_report, orient="index", columns=["outlier_count"]
        )

    # =========================================================
    # SKEWNESS REPORT
    # =========================================================

    def skewness_report(self):
        """Prints a report of skewness for each numerical column."""

        skew_report = {}

        for col in self.numeric_cols:
            skew_report[col] = skew(self.df[col].dropna())

        return pd.DataFrame.from_dict(
            skew_report, orient="index", columns=["skewness"]
        ).sort_values(by="skewness", ascending=False)

    # =========================================================
    # LOG TRANSFORM SUGGESTIONS
    # =========================================================

    def log_transform_candidates(self, threshold=1):
        """Identifies numerical columns that are candidates for log transformation based on skewness."""

        candidates = []

        for col in self.numeric_cols:
            s = skew(self.df[col].dropna())

            if s > threshold:
                candidates.append({"feature": col, "skewness": round(s, 3)})

        return pd.DataFrame(candidates).sort_values(by="skewness", ascending=False)

    # =========================================================
    # CORRELATION MATRIX
    # =========================================================

    def correlation_heatmap(self):
        """Plots a correlation heatmap for numerical columns."""

        corr = self.df[self.numeric_cols].corr()

        plt.figure(figsize=(20, 20))

        ax = sns.heatmap(corr, annot=True, fmt=".3f", linewidth=0.5, cmap="coolwarm")
        ax.xaxis.tick_top()
        # annotations text size
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14, rotation=0)

        plt.title(
            "Correlation Heatmap for Numerical Features",
            fontsize=20,
            weight="bold",
            pad=20,
        )
        plt.tight_layout()

        plt.show()

    # =========================================================
    # DISTRIBUTION PLOTS
    # =========================================================

    def distribution_plots(self):
        """Plots distribution plots for each numerical column."""

        for col in self.numeric_cols:
            plt.figure(figsize=(16, 9))
            fig, ax = plt.subplots()

            sns.histplot(
                self.df[col],
                bins="auto",
                kde=True,
                line_kws={"linewidth": 1, "color": "red"},
                ax=ax,
            )

            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

            plt.tight_layout()

            plt.show()

    # =========================================================
    # BOXPLOTS
    # =========================================================

    def boxplots(self):
        """Plots boxplots for each numerical column."""

        for col in self.numeric_cols:
            plt.figure(figsize=(16, 9))
            fig, ax = plt.subplots()

            sns.boxplot(x=self.df[col], ax=ax)

            ax.set_title(f"Boxplot: {col}")
            ax.set_xlabel(col)
            plt.tight_layout()

            plt.show()

    # =========================================================
    # TARGET RELATIONSHIPS
    # =========================================================

    def target_relationships(self):
        """Plots scatter plots to visualize relationships between each numerical column and the target."""

        if self.target is None:
            print("No target column specified.")
            return

        for col in self.numeric_cols:
            if col == self.target:
                continue

            plt.figure(figsize=(16, 9))
            fig, ax = plt.subplots()

            sns.scatterplot(data=self.df, x=col, y=self.target, ax=ax)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{self.target} vs {col}")
            plt.tight_layout()

            plt.show()

    # =========================================================
    # FULL REPORT
    # =========================================================

    def run_all(self):
        """Runs all EDA functions and displays the reports."""

        self.overview()

        self.memory_report()

        print("\n")
        self.summary_stats()

        print("\n")
        print("=" * 60)
        print("MISSING VALUE REPORT")
        print("=" * 60)
        print(self.missing_report())

        print("\n")
        print("=" * 60)
        print("CARDINALITY REPORT")
        print("=" * 60)
        print(self.cardinality_report())

        print("\n")
        print("=" * 60)
        print("OUTLIER REPORT")
        print("=" * 60)
        print(self.detect_outliers_iqr())

        print("\n")
        print("=" * 60)
        print("SKEWNESS REPORT")
        print("=" * 60)
        print(self.skewness_report())

        print("\n")
        print("=" * 60)
        print("LOG TRANSFORM CANDIDATES")
        print("=" * 60)
        print(self.log_transform_candidates())

        print("\n")
        print("=" * 60)
        print("Showing correlation heatmap...")
        print("=" * 60)
        self.correlation_heatmap()

        print("\n")
        print("=" * 60)
        print("Showing distribution plots...")
        print("=" * 60)
        self.distribution_plots()

        print("\n")
        print("=" * 60)
        print("Showing boxplots...")
        print("=" * 60)
        self.boxplots()

        print("\n")
        print("=" * 60)
        print("Showing target vs feature relationships...")
        print("=" * 60)
        self.target_relationships()

        print("\n")
        print("=" * 60)
        print("Task complete! All EDA steps have been executed.")
        print("=" * 60)
