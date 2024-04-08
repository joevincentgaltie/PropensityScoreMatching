from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from PropensityScoreMatching import utils

utils.set_graph_style()


class DoubleDiff(ABC):
    """DoubleDiff
    This class helps to plot and generate confidence intervals through bootstrapping on a time serie of two or more groups.
    It computes the mean, and confidence interval of each group at each time.

    """

    def __init__(
        self,
        var_group_distinction: str,
        var_time_series: str,
        color_palette: list = ["#096c45", "#d69a00", "#9f0025", "#e17d18"],
    ):
        """
        args :

        - var_group_distinction (str) : col name that enables to distinguish groups
        - var_time_series(str) : col name that gives the time period and that will be the x axis of the plot.
        - color_palette (list) : list of colors for the plot
        """

        self.var_group_distinction = var_group_distinction
        self.var_time_series = var_time_series
        self.color_palette = color_palette

    def _get_var_bootstrapped(self, df: pd.DataFrame, var: str):
        """Computes the confidence interval bootstrapped for a var at each time period and for each group.

        args :

        - df (pd.DataFrame) : DataFrame of time series, with a column that helps to dsitinguish groups, and a column giving the time period to group on.
        - var (str) : col name thta needs to be analyzed.
        Returns:
            pd.DataFrame: DataFrame giving mean, and confidence intervals for var at each period for each group.
        """

        df_bootstrapped = df.groupby(
            [self.var_group_distinction, self.var_time_series]
        ).agg(
            {
                var: [
                    ("Moyenne", "mean"),
                    (
                        "inf",
                        lambda x: utils.bootstrap_mean_confidence_interval(
                            data=x.dropna()
                        )[0],
                    ),
                    (
                        "sup",
                        lambda x: utils.bootstrap_mean_confidence_interval(
                            data=x.dropna()
                        )[1],
                    ),
                ]
            }
        )
        df_bootstrapped.columns = df_bootstrapped.columns.droplevel()
        df_bootstrapped.reset_index(inplace=True)

        return df_bootstrapped

    def plot_var_bootstrapped(self, df: pd.DataFrame, var: str):
        """Plots the mean and its confidence interval for var over time for each group.

        args :

        - df (pd.DataFrame) : DataFrame of time series, with a column that helps to dsitinguish groups, and a column giving the time period to group on.
        - var (str) : col name that needs to be analyzed.

        """

        df_bootstrapped = self._get_var_bootstrapped(df=df, var=var)

        plt.figure()

        for group, color in zip(
            df_bootstrapped[self.var_group_distinction].unique(), self.color_palette
        ):
            sns.lineplot(
                data=df_bootstrapped[
                    df_bootstrapped[self.var_group_distinction] == group
                ],
                x=self.var_time_series,
                y="Moyenne",
                label=var + "_mean_treated",
                color=color,
            )
            sns.lineplot(
                data=df_bootstrapped[
                    df_bootstrapped[self.var_group_distinction] == group
                ],
                x=self.var_time_series,
                y="inf",
                linestyle="--",
                color=color,
            )
            sns.lineplot(
                data=df_bootstrapped[
                    df_bootstrapped[self.var_group_distinction] == group
                ],
                x=self.var_time_series,
                y="sup",
                linestyle="--",
                color=color,
            )
            plt.fill_between(
                x=df_bootstrapped[df_bootstrapped[self.var_group_distinction] == group][
                    self.var_time_series
                ],
                y1=df_bootstrapped[
                    df_bootstrapped[self.var_group_distinction] == group
                ]["inf"],
                y2=df_bootstrapped[
                    df_bootstrapped[self.var_group_distinction] == group
                ]["sup"],
                color=color,
                alpha=0.3,
            )

        plt.legend(bbox_to_anchor = (0.5, -0.15), ncol = 2)
        plt.title(f"Evol of {var}")
        plt.show()
