import pandas as pd
import numpy as np
from pprint import pprint


def typecast_dataframe(
    df,
    int_cols=None,
    float_cols=None,
    str_cols=None,
    bool_cols=None,
    category_cols=None,
    datetime_cols=None,
    datetime_format=None,
    errors="coerce",
    verbose=True,
):
    """
    Robust DataFrame type casting with validation and reporting.
    Useful for ensuring data integrity and understanding the impact of type conversions,
    especially when dealing with messy or inconsistent datasets.

    Args
    ----
    df : pd.DataFrame
        Input DataFrame to be typecasted.
    int_cols : list of str, optional
        Columns to convert to integer (nullable Int64).
    float_cols : list of str, optional
        Columns to convert to float.
    str_cols : list of str, optional
        Columns to convert to string (pandas StringDtype).
    bool_cols : list of str, optional
        Columns to convert to boolean (pandas BooleanDtype).
    category_cols : list of str, optional
        Columns to convert to category.
    datetime_cols : list of str, optional
        Columns to convert to datetime.
    datetime_format : str, optional
        Datetime format string for parsing datetime columns (e.g. "%Y-%m-%d").
    errors : {'raise', 'coerce', 'ignore'}, default 'coerce'
        How to handle conversion errors:
        - 'raise': Raise an exception on conversion failure.
        - 'coerce': Set invalid parsing to NaN/NA.
        - 'ignore': Return the original data for invalid parsing.

    Returns
    -------
    df_casted : pd.DataFrame
    report : dict

    Usage
    -----
    typecast_config = {
    "int_cols": [],
    "float_cols": [],
    "str_cols": [],
    "bool_cols": [],
    "category_cols": ['src_asn', 'dst_asn'],
    "datetime_cols": []
    }

    df_casted, typecast_report = typecast_dataframe(
        df,
        errors='coerce',
        verbose=True,
        **typecast_config
        )
    """

    df = df.copy()

    int_cols = int_cols or []
    float_cols = float_cols or []
    str_cols = str_cols or []
    bool_cols = bool_cols or []
    category_cols = category_cols or []
    datetime_cols = datetime_cols or []

    requested_cols = (
        int_cols + float_cols + str_cols + bool_cols + category_cols + datetime_cols
    )

    missing_cols = [col for col in requested_cols if col not in df.columns]

    report = {
        "missing_columns": missing_cols,
        "dtype_changes": {},
        "conversion_failures": {},
    }

    # -----------------------------
    # Strip whitespace from object columns
    # -----------------------------
    obj_cols = df.select_dtypes(include=["object"]).columns

    for col in obj_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # -----------------------------
    # Save original dtypes
    # -----------------------------
    original_dtypes = df.dtypes.astype(str).to_dict()

    # -----------------------------
    # Integer columns
    # -----------------------------
    for col in int_cols:
        if col not in df.columns:
            continue

        original_nulls = df[col].isna().sum()

        numeric = pd.to_numeric(df[col], errors=errors)

        failures = numeric.isna().sum() - original_nulls

        report["conversion_failures"][col] = int(max(failures, 0))

        df[col] = numeric.astype("Int64")

    # -----------------------------
    # Float columns
    # -----------------------------
    for col in float_cols:
        if col not in df.columns:
            continue

        original_nulls = df[col].isna().sum()

        numeric = pd.to_numeric(df[col], errors=errors)

        failures = numeric.isna().sum() - original_nulls

        report["conversion_failures"][col] = int(max(failures, 0))

        df[col] = numeric

    # -----------------------------
    # String columns
    # -----------------------------
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # -----------------------------
    # Boolean columns
    # -----------------------------
    bool_map = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False,
        "t": True,
        "f": False,
    }

    for col in bool_cols:
        if col not in df.columns:
            continue

        def convert_bool(x):
            if pd.isna(x):
                return pd.NA

            if isinstance(x, bool):
                return x

            key = str(x).strip().lower()

            return bool_map.get(key, pd.NA)

        converted = df[col].apply(convert_bool)

        report["conversion_failures"][col] = int(
            converted.isna().sum() - df[col].isna().sum()
        )

        df[col] = converted.astype("boolean")

    # -----------------------------
    # Category columns
    # -----------------------------
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # -----------------------------
    # Datetime columns
    # -----------------------------
    for col in datetime_cols:
        if col not in df.columns:
            continue

        original_nulls = df[col].isna().sum()

        converted = pd.to_datetime(
            df[col],
            format=datetime_format,
            errors=errors,
        )

        failures = converted.isna().sum() - original_nulls

        report["conversion_failures"][col] = int(max(failures, 0))

        df[col] = converted

    # -----------------------------
    # Dtype report
    # -----------------------------
    new_dtypes = df.dtypes.astype(str).to_dict()

    for col in df.columns:
        if original_dtypes[col] != new_dtypes[col]:
            report["dtype_changes"][col] = {
                "from": original_dtypes[col],
                "to": new_dtypes[col],
            }

    if verbose:
        print("=== TYPE CAST REPORT ===\n")

        if missing_cols:
            pprint(f"\nMissing columns: {missing_cols}")

        print("Dtype changes:\n")
        for col, change in report["dtype_changes"].items():
            pprint(f"  {col}: {change['from']} -> {change['to']}")

        print("Conversion failures:\n")
        for col, count in report["conversion_failures"].items():
            if count > 0:
                pprint(f"  {col}: {count}")
            else:
                pprint(f"  {col}: None")

    return df, report
