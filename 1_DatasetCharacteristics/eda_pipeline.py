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

from pprint import pprint

import pandas as pd
import numpy as np
from scipy.stats import skew

# -------------- Plotting libraries  & in depth settings--------------
import matplotlib.pyplot as plt
import seaborn as sns

# a rc param dict to set the default figure size and other parameters for seaborn plots
plot_params = {
    "figure.figsize": (15, 6),  # Set default figure size for all plots (Ultrawide)
    "figure.dpi": 150,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.shadow": True,
    "grid.alpha": 0.25,  # Set grid transparency for better visibility
    "savefig.bbox": "tight",  # Ensure tight layout when saving figures
}


sns.set_theme(
    style="whitegrid",
    context="notebook",
    font="sans-serif",
    font_scale=1.0,
    rc=plot_params,
)

# sns.set_theme(style="whitegrid", context="talk", font_scale=1)
# sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
# sns.set_theme(style="ticks", context="notebook", font_scale=1.2)
# sns.set_theme(style="white", context="poster", font_scale=1)

# ---------------------- Icons --------------------------------
# EDA UI Icons
ICON_START = "🚀"
ICON_ALERT = "🚨"
ICON_DONE = "✅"
ICON_DATA_CLEANING = "🧹"
ICON_DATASET_OVERVIEW = "📋"
ICON_NUMERICAL_SUMMARY = "🔢"
ICON_CATEGORICAL_OVERVIEW = "🔠"
ICON_BAR_CHART = "📊"
ICON_LINE_CHART = "📈"
ICON_CORRELATION_MATRIX = "🌡️"
ICON_FEATURE_ENGINEERING = "🛠️"
ICON_TRAIN_TEST_SPLIT = "✂️"
ICON_KEY_INSIGHTS = "💡"
ICON_EXPORT_DATA = "💾"
# ------------------------------------------------------------

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

        print("+" * 80)
        print(
            " " * 20
            + f" {ICON_START} EDA Script for Initial Analysis Started !"
            + " " * 20
        )
        print("+" * 80)

        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print(f"Shape: {self.df.shape}")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")

        print("\nColumns:")
        print("-" * 30)
        pprint(self.df.columns.tolist())

        print("\nData Types:")
        print(
            "#"
            + "-" * 30
            + f" {ICON_ALERT} Check for Typecast Issues ! {ICON_ALERT}"
            + "-" * 30
            + "#\n"
        )
        pprint(self.df.dtypes)

        print("\n" + ICON_ALERT + " Missing Values:")
        print("-" * 30)
        pprint(self.df.isnull().sum().sort_values(ascending=False))

        print("\n" + ICON_ALERT + " Duplicate Rows:")
        print("-" * 30)
        pprint(self.df.duplicated().sum())

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
        print(f" {ICON_NUMERICAL_SUMMARY} NUMERICAL SUMMARY {ICON_NUMERICAL_SUMMARY} ")
        print("=" * 60)

        pprint(self.df[self.numeric_cols].describe().T)

    # =========================================================
    # MISSING VALUE REPORT
    # =========================================================

    def missing_report(self):
        """Prints a report of missing values for each column."""

        missing = pd.DataFrame(
            {
                "missing_count": self.df.isnull().sum(),
                "missing_percent (%)": self.df.isnull().mean() * 100,
            }
        )

        missing = missing.sort_values(by="missing_percent (%)", ascending=False)

        return missing

    # =========================================================
    # HIGH CARDINALITY REPORT
    # =========================================================

    def cardinality_report(self):
        """Prints unique values and offers typecasting hints based on ratio."""
        report = {}

        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            unique_ratio = unique_count / len(self.df)

            if unique_count <= 2:
                suggestion = "bool flag, consider bool typecast"
            elif unique_ratio < 0.1:
                suggestion = "consider typecast to category"
            else:
                suggestion = "High cardinality, object or alternative encoding method"

            report[col] = {
                "unique_values": unique_count,
                "cardinality_percent (%)": unique_ratio * 100,
                "suggestion": suggestion,
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
        ).sort_values(by="outlier_count", ascending=False)

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
        """Plots distribution plots for high cardinality *(> 10) numerical columns."""

        # we use a cardinality logic to find suitable columns for dist plots
        suitable_cols = [
            col for col in self.numeric_cols if self.df[col].nunique() > 10
        ]

        # also include categorical cols with the same logic
        # suitable_cols.extend(
        #    [col for col in self.categorical_cols if self.df[col].nunique() > 10]
        # )

        pprint(f"Showing Distribution Plots for : {', '.join(suitable_cols)}")

        for col in suitable_cols:
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
    # COUNTPLOTS
    # =========================================================
    def countplots(self):
        """Plots countplots for each categorical column."""

        # Use cardinality logic to find suitable columns
        suitable_cols = [
            col for col in self.numeric_cols if self.df[col].nunique() <= 10
        ]

        # Include categorical cols
        suitable_cols.extend(
            [col for col in self.categorical_cols if self.df[col].nunique() <= 10]
        )

        print(f"Showing Count Plots for : {', '.join(suitable_cols)}")

        for col in suitable_cols:
            fig, ax = plt.subplots()

            # Cast the entire column to string to prevent None/Bool sorting crashes
            safe_series = self.df[col].astype(str)
            pallete = sns.color_palette("rocket", n_colors=safe_series.nunique())
            # Passing x and hue directly as series bypasses the need for the data argument
            sns.countplot(
                x=safe_series,
                order=safe_series.value_counts().index,
                ax=ax,
                hue=safe_series,
                legend=False,
                palette=pallete,
            )

            # Show the count values on top of the bars
            for p in ax.patches:
                count = int(p.get_height())
                ax.annotate(
                    f"{count}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_title(f"Countplot: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=35)

            plt.tight_layout()
            plt.show()

    # =========================================================
    # BOXPLOTS
    # =========================================================

    def boxplots(self):
        """Plots box & whisker plots for each numerical column with statistical metrics in a legend text box."""
        # we use a cardinality logic to find suitable columns for boxplots
        suitable_cols = [
            col for col in self.numeric_cols if self.df[col].nunique() > 10
        ]

        pprint(f"Showing Box Plots for : {', '.join(suitable_cols)}")

        for col in suitable_cols:
            # Drop missing values for clean statistical calculation
            col_data = self.df[col].dropna()
            if col_data.empty:
                continue

            fig, ax = plt.subplots()
            sns.boxplot(x=col_data, ax=ax)

            # Calculate key metrics
            stats = col_data.describe()
            q1 = stats["25%"]
            median = stats["50%"]
            q3 = stats["75%"]

            # Calculate whiskers limits matching seaborn's default (IQR * 1.5)
            iqr = q3 - q1
            lower_whisker = col_data[col_data >= (q1 - 1.5 * iqr)].min()
            upper_whisker = col_data[col_data <= (q3 + 1.5 * iqr)].max()

            # Find the percentile of the upper whisker to show in the text box
            upper_whisker_percentile = (
                col_data[col_data <= upper_whisker].shape[0] / col_data.shape[0]
            ) * 100

            # num of outliers beyond the upper whisker
            outliers_count = col_data[col_data > upper_whisker].shape[0]
            outliers_percent = (outliers_count / col_data.shape[0]) * 100

            # num of outliers beyond the lower whisker
            lower_outliers_count = col_data[col_data < lower_whisker].shape[0]
            lower_outliers_percent = (lower_outliers_count / col_data.shape[0]) * 100

            # Format the metrics text block
            stats_text = (
                f"  Upper Whisker:  {upper_whisker:.2f} ({upper_whisker_percentile:.1f}th percentile)\n"
                f"  Outliers beyond upper whisker: {outliers_count} ({outliers_percent:.2f}%)\n"
                f"  Q3:   {q3:.2f}\n"
                f"  Med:  {median:.2f}\n"
                f"  Q1:   {q1:.2f}\n"
                f"  Lower Whisker:  {lower_whisker:.2f}\n"
                f"  Outliers below lower whisker: {lower_outliers_count} ({lower_outliers_percent:.2f}%)\n"
            )

            # Place the text box inside the plot using relative axes coordinates (0 to 1)
            # loc="upper right" equivalent using transform=ax.transAxes
            # use loc = "best"
            ax.legend(
                title="Statistical Summary",
                labels=[stats_text],
                loc="best",
                fontsize=11,
            )

            """
            ax.text(
                x=0.95,
                y=0.90,
                s=stats_text,
                transform=ax.transAxes,
                fontsize=11,
                fontfamily="monospace",  # Monospace ensures numbers line up perfectly
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8
                ),
            )
            """
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
        print("=" * 80)
        print(f"{ICON_ALERT} MISSING VALUE REPORT {ICON_ALERT}")
        print("=" * 80)
        print(self.missing_report())

        print("\n")
        print("=" * 80)
        print(
            f"{ICON_CATEGORICAL_OVERVIEW} CARDINALITY REPORT {ICON_CATEGORICAL_OVERVIEW}"
        )
        print("=" * 80)
        print(self.cardinality_report())

        print("\n")
        print("=" * 80)
        print(f"{ICON_KEY_INSIGHTS} OUTLIER REPORT {ICON_KEY_INSIGHTS}")
        print("=" * 80)
        print(self.detect_outliers_iqr())

        print("\n")
        print("=" * 80)
        print(f"{ICON_KEY_INSIGHTS} SKEWNESS REPORT {ICON_KEY_INSIGHTS}")
        print("=" * 80)
        print(self.skewness_report())

        print("\n")
        print("=" * 80)
        print(f"{ICON_KEY_INSIGHTS} LOG TRANSFORM CANDIDATES {ICON_KEY_INSIGHTS}")
        print("Benefitial for :\n")
        print(
            "- Linear Regression, Ridge, Lasso, & ElasticNet ;\n "
            "- Logistic Regression & Support Vector Machines (SVM),\n "
            "- Neural Networks, KNN & K-Means"
        )
        print("=" * 80)
        print(self.log_transform_candidates())

        print("+" * 80)
        print(" " * 20 + " Printing plots..." + " " * 20)
        print("+" * 80)

        print("\n")
        print("=" * 80)
        print(
            " " * 20
            + f"{ICON_CORRELATION_MATRIX} Showing Correlation Heatmap... "
            + " " * 20
        )
        print("=" * 80)
        self.correlation_heatmap()

        print("\n")
        print("=" * 80)
        print(" " * 20 + f"{ICON_BAR_CHART} Showing Distribution Plots... " + " " * 20)
        print("=" * 80)
        self.distribution_plots()

        print("\n")
        print("=" * 80)
        print(" " * 20 + f"{ICON_BAR_CHART} Showing Count Plots... " + " " * 20)
        print("=" * 80)
        self.countplots()

        print("\n")
        print("-" * 80)
        print(
            " " * 20 + f"{ICON_BAR_CHART} Showing Box and Whisker Plots... " + " " * 20
        )
        print("-" * 80)
        self.boxplots()

        print("\n")
        print("=" * 80)
        print(
            " " * 20
            + f"{ICON_LINE_CHART} Showing Target vs Feature Relationships... "
            + " " * 20
        )
        print("=" * 80)
        self.target_relationships()

        print("\n")
        print("+" * 80)
        print(
            " " * 20
            + f"{ICON_DONE} Task complete! All EDA steps have been executed."
            + " " * 20
        )
        print("+" * 80)
