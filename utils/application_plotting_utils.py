import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def expand_rmse_crps_and_coverages(df: pd.DataFrame, include_coverage=False):
    n_obs = df["n_test_obs_vals"].loc[0]
    df[[f"rmse_{n}" for n in n_obs]] = pd.DataFrame(
        df["rmse_vals"].tolist(), index=df.index
    )
    df[[f"crps_{n}" for n in n_obs]] = pd.DataFrame(
        df["crps_vals"].tolist(), index=df.index
    )
    if include_coverage:
        df[[f"coverage_{n}" for n in n_obs]] = pd.DataFrame(
            df["coverage_vals"].tolist(), index=df.index
        )
    return df


def import_df(storage, file_path: str, type: str, include_coverage=True):
    df = storage.load_results(file_path)
    df = expand_rmse_crps_and_coverages(df, include_coverage)
    df["type"] = type
    return df


def create_NF_F_diff_df(
    df_NF: pd.DataFrame, df_F: pd.DataFrame, type: str
) -> pd.DataFrame:
    n_obs = df_NF["n_test_obs_vals"].loc[0]

    df = pd.DataFrame()
    df["n_test_obs_vals"] = df_NF["n_test_obs_vals"]
    df["type"] = type
    for n in n_obs:
        df[f"rmse_diff_{n}"] = df_F[f"rmse_{n}"] - df_NF[f"rmse_{n}"]
        df[f"crps_diff_{n}"] = df_F[f"crps_{n}"] - df_NF[f"crps_{n}"]
    return df


def print_freqs_and_penalties(df: pd.DataFrame):
    print(
        df[
            [
                "n_freqs_logkappa",
                "n_freqs_logsigma",
                "n_freqs_v",
                "penalty_range",
                "penalty_sd",
                "penalty_anisotropy",
            ]
        ].loc[0]
    )


def print_mean_and_std_for_type(df: pd.DataFrame, col: str):
    mean_std_df = df.groupby("type")[col].agg(["mean", "std"])
    print(mean_std_df)
