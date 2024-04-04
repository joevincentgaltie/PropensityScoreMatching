import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

""" une classe abstraite qui donne un formalisme pour tous les matchers"""


class Matcher(ABC):
    def __init__(self, var_treatment: str, ratio: int = 1):
        """__init__


        Args:
            ratio (int): ratio of control individuals vs treated ones
            var_treatment (str): var name that indicates whether an individual has been treated or not. This var takes 0 or 1.
        """
        self.ratio = ratio
        assert self.ratio >= 1

        self.var_treatment = var_treatment
        pass

    @abstractmethod
    def match(df1):
        pass


"""La vraie classe"""


class PropensityScoreMatcher(Matcher):
    """

    ```python

    from PropensityScoreMatching import PropensityScoreMatcher



    ```
    """

    def __init__(
        self,
        var_treatment: str,
        id_var: str,
        ratio: int = 1,
        caliper=0.15,
        random_state=42,
    ):
        """



        Args:
            ratio (int): ratio of control individuals vs treated ones
            var_treatment (str): var name that indicates whether an individual has been treated or not. This var takes 0 or 1.
            time_series_var (str) : var name of the time serie variable (eg. year, month day...). If the df is not a time serie var, var needs to be set on ""
            id_var (str) : name of the var giving a id to identify conssitently any individual over time
            caliper (float) : maximum spread for matching on propensity score, default is 0.15 like in literature
            cylinder (bool) : if True, a treated individual will be match with a control that has a presence over the exact same period. -- only concerns time series
        """

        super().__init__(ratio=ratio, var_treatment=var_treatment)

        self.random_state = random_state

        self.id_var = id_var
        self.caliper = caliper
        assert (self.caliper >= 0) & (self.caliper <= 1)

    # Logit

    def fit_logit_on_df(self, df: pd.DataFrame, var_logit: list) -> pd.DataFrame:
        """fit_logit_on_df

        fits a logit using statsmodel.api.formula on the dataframe and adds the propensity score to the dataframe.
        It also prints the summary of the logit.

        Args:
            df (pd.DataFrame): _description_
            var_logit (list): _description_

        Returns:
            pd.DataFrame: _description_
        """

        model = smf.logit(
            self.var_treatment + " ~ " + f" + ".join([var for var in var_logit]),
            data=df,
        )
        result = model.fit()
        df["propensity_score"] = result.predict(df)
        print(result.summary())
        print(
            "Propensity scores were added to the dataframe in column 'propensity_score'"
        )
        return df

    def check_plot_common_support(self, df):
        """check_plot_common_support

        Checks graphically the common support of the propensity score between the treated and control group by plotting the distribution of the propensity score for each group.

        Args:
            df (_type_): df after the logit has been fitted and the propensity score has been added to the dataframe.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.histplot(
            df,
            x="propensity_score",
            hue=self.var_treatment,
            ax=ax,
            bins=1000,
            alpha=0.7,
            kde=True,
        )
        plt.xlabel("Propensity score")
        plt.ylabel("Density")
        plt.show()
        pass

    ### Matching

    def _get_closest_neighbors(
        self,
        df: pd.DataFrame,
        valeur_propensity: float,
        n_neighbors: int,
        random_state: int,
    ) -> list:
        """_get_closest_neighbors
        For a given propensity score, this function will return the closest individuals in the dataframe within the caliper.
        It is a draw with replacement !!

        Args:
            df (pd.DataFrame): df on which logit has been fitted and propensity score has been added
            valeur_propensity (float): propensity score of the individual to match
            n_neighbors (int): number of neighbors to return (eg ratio)
            random_state (int): seed for random

        Returns:
            index_closest_within_caliper (List): List of index of the closest individuals within the caliper.
        """

        diff = (df["propensity_score"] - valeur_propensity).abs()
        random.seed(random_state)
        keep_opt = random.choice(["first", "last"])
        index_closest = diff.nsmallest(n_neighbors, keep=keep_opt).index.to_list()
        index_closest_within_caliper = [
            idx
            for idx in index_closest
            if abs(df["propensity_score"][idx] - valeur_propensity) <= self.caliper
        ]
        return index_closest_within_caliper

    def match(
        self, df: pd.DataFrame, exact_matching: Union[List, None] = None
    ) -> pd.DataFrame:
        """lorem ipsum

        Args:
            var (type):

        Return:

        """

        assert df[self.var_treatment].nunique() == 2

        # result = self._fit_logit(df, var_logit)        # compute the logit
        # df["propension_score"] = result.predict(df)    #add logit score to dataframe
        # self._check_common_support(df)

        # match
        self._matching_indices = {}
        for idx1, row in tqdm(
            df.iterrows(), desc="Processing to matching row by row", total=df.shape[0]
        ):
            if row[self.var_treatment] == 1:
                tmp_df_to_match_on = df[df[self.var_treatment] == 0]

                if exact_matching:
                    tmp_df_to_match_on = tmp_df_to_match_on[
                        (tmp_df_to_match_on[exact_matching] == row[exact_matching]).all(
                            axis=1
                        )
                    ]

                closest = self._get_closest_neighbors(
                    tmp_df_to_match_on,
                    valeur_propensity=row["propensity_score"],
                    n_neighbors=self.ratio,
                    random_state=self.random_state,
                )

                if closest:
                    self._matching_indices[row[self.id_var]] = (
                        df.loc[closest, self.id_var].unique().tolist()
                    )  # get id_var at index closest

        # add matched index to dataframe
        df["matched_id"] = df[self.id_var].map(self._matching_indices)
        list_counterfactual = df["matched_id"].explode().unique().tolist()

        # filter df to keep only matched rows and treated rows
        df.loc[
            df[self.id_var].isin(
                list_counterfactual + list(self._matching_indices.keys())
            ),
            "to_keep_after_matching",
        ] = 1
        df["to_keep_after_matching"].fillna(0, inplace=True)
        return df

    # Analysis

    def _mean_difference_(self, df: pd.DataFrame, var_to_plot: list):
        """_mean_difference_ _summary_

        Args:
            df (pd.DataFrame): _description_
            var_to_plot (list): _description_

        Returns:
            _type_: _description_
        """

        assert "to_keep_after_matching" in df.columns
        assert "treatment" in df.columns

        tmp_left = df.groupby("treatment")[var_to_plot].mean().T
        tmp_left = pd.DataFrame(
            100
            * (abs(tmp_left[1] - tmp_left[0]) / abs(tmp_left[1])).rename(
                "gap between mean of treated group and others before matching"
            )
        )
        tmp_right = (
            df[df["to_keep_after_matching"] == 1]
            .groupby("treatment")[var_to_plot]
            .mean()
            .T
        )
        tmp_right = pd.DataFrame(
            100
            * (abs(tmp_right[1] - tmp_right[0]) / abs(tmp_right[1])).rename(
                "gap between mean of treated group and others after matching"
            )
        )
        tmp = pd.concat([tmp_left, tmp_right], axis=1)
        return tmp

    def _plot_mean_difference_(self, df: pd.DataFrame, var_to_plot: list):
        """_plot_mean_difference_ _summary_

        Args:
            df (pd.DataFrame): _description_
            var_to_plot (list): _description_
        """
        tmp_2 = self._mean_difference_(df, var_to_plot)
        tmp_2.plot(kind="bar", figsize=(10, 6))
        formatter = FuncFormatter(lambda x, _: "{:.0%}".format(x / 100))
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.ylabel("Percentage (%)")
        plt.title(
            "mean difference between treated and control before and after matching"
        )
        plt.show()

    def report(self, df: pd.DataFrame, var_to_plot: list):
        """_summary_"""
        treated = df[df[self.var_treatment] == 1][self.id_var].nunique()
        control = df[df[self.var_treatment] == 0][self.id_var].nunique()

        treated_after_matching = df[
            (df[self.var_treatment] == 1) & (df["to_keep_after_matching"] == 1)
        ][self.id_var].nunique()
        control_after_matching = df[
            (df[self.var_treatment] == 0) & (df["to_keep_after_matching"] == 1)
        ][self.id_var].nunique()
        # this information in  a df with columns before, and after matching

        table = pd.DataFrame(
            {
                "Before matching": [treated, control],
                "After matching": [treated_after_matching, control_after_matching],
                "Attrition": [
                    100 * (1 - treated / treated_after_matching),
                    100 * (1 - control / control_after_matching),
                ],
            },
            index=["Treated", "Control"],
        )
        print(table)
        print(
            f"Before matching, there were {treated} treated individuals and {control} control individuals"
        )
        print(
            f"After matching, there are {treated_after_matching} treated individuals and {control_after_matching} control individuals"
        )
        self._plot_mean_difference_(df, var_to_plot)

    # Analysis

    def get_dataframe_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """The match function returns a dataframe enriched with the matched_id. It identifies the individuals that are part of the treatment group and the control group.
        This function will return a dataframe that is ready for analysis. More precisely, it will return a dataframe that contains the treated individuals and the matched ones duplicated as many times as they have been matched.

        args :
            df (pd.DataFrame : dataframe that has been matched.

        Returns:
            pd.DataFrame : _description_
        """

        assert "matched_id" in df.columns
        assert "to_keep_after_matching" in df.columns

        treated_list = (
            df[(df[self.var_treatment] == 1)][self.id_var].value_counts().to_dict()
        )
        matched_list = (
            df[(df[self.var_treatment] == 1)]["matched_id"]
            .explode()
            .value_counts()
            .to_dict()
        )
        len(list(set(treated_list.keys()) & set(matched_list.keys()))) == 0

        counter = treated_list.copy()
        counter.update(matched_list)

        df_for_analysis = []
        for k, v in counter.items():
            df_for_analysis.append(pd.concat([df[df[self.id_var] == k]] * v, axis=0))
        df_for_analysis = pd.concat(df_for_analysis, axis=0)
        return df_for_analysis


# cette commande renvoie la docstring de la classe PropensityScoreMatcher


# def get_closest_neighbor(serie :pd.Series(), valeur : float, n:int, random_state) -> int :
#     diff = (serie-valeur).abs()
#     random.seed(random_state)
#     keep_opt = random.choice(['first','last'])
#     index_closest = diff.nsmallest(n, keep = keep_opt).index.to_list()
#     return index_closest


class PropensityScoreMatcherTS(PropensityScoreMatcher):

    """ """

    def __init__(
        self,
        var_treatment: str,
        id_var: str,
        ratio: int = 1,
        time_series_var: Union[str, None] = None,
        caliper=0.15,
        cylinder=False,
        random_state=42,
    ):
        """
        Args:
            ratio (int): ratio of control individuals vs treated ones
            var_treatment (str): var name that indicates whether an individual has been treated or not. This var takes 0 or 1.
            time_series_var (str) : var name of the time serie variable (eg. year, month day...). If the df is not a time serie var, var needs to be set on ""
            id_var (str) : name of the var giving a id to identify conssitently any individual over time
            caliper (float) : maximum spread for matching on propensity score, default is 0.15 like in literature
            cylinder (bool) : if True, a treated individual will be match with a control that has a presence over the exact same period. -- only concerns time series
        """

        super().__init__(
            ratio=ratio,
            var_treatment=var_treatment,
            id_var=id_var,
            random_state=random_state,
            caliper=caliper,
        )

        self.time_series_var = time_series_var
        self.cylinder = cylinder

    # Preprocessing of the time series df

    def _write_treatment_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """_write_treatment_period


        This function is only called for time series. This function will indicate it is t+0 t+1 t+2 etc for each row cocnerning an individual treated at time t

        Args:
            df (pd.DataFrame): DataFrame that requires for each individual to have only 1 treatment over the period.

        Returns:
            pd.DataFrame: DataFrame enriched with :
                        - self.time_series_var + "_treatment" that indicated the time of treatment
                        - treatment_period that indicates time to treatment
                        - treated_over_time that indicates if a individual is part of the treatment group or control one.



        """

        year_treatment_per_id = df[df[self.var_treatment] == 1][
            [self.id_var, self.time_series_var]
        ].drop_duplicates()  # get the year of treatment for each individual
        assert (
            year_treatment_per_id[self.id_var].nunique()
            == year_treatment_per_id.shape[0]
        )  # asserts that only one time of treatment exists by individual
        df = df.merge(
            year_treatment_per_id,
            on=self.id_var,
            how="left",
            suffixes=("", "_treatment"),
        )  # merge the year of treatment to the dataframe

        for idx, row in tqdm(
            df.iterrows(), desc="Processing to add treatment period", total=df.shape[0]
        ):
            if pd.isnull(
                row[self.time_series_var + "_treatment"]
            ):  # if the individual has never been treated (eligible to be control), it is set to nan
                df.loc[idx, "treatment_period"] = np.nan
            else:  # treatment_period is the difference between the time of treatment and the time of the row
                df.loc[idx, "treatment_period"] = (
                    row[self.time_series_var] - row[self.time_series_var + "_treatment"]
                )

        df["treated_over_time"] = df[self.time_series_var + "_treatment"].apply(
            lambda x: 1 if not np.isnan(x) else 0
        )  # will help to filter the dataframe for matching
        return df

    def _identify_period_of_presence(self, df: pd.DataFrame) -> dict:
        """_identify_years_of_presence

        In order to process to cylinder, that function stores the exact period of presence of the indviduals.

        Args:
            df (pd.DataFrame): Only concerns df that have a time_serie_var.

        Returns:
            dict: its format is {id : List(time of presence), ... }
        """
        period_of_presence = {}
        for idx in df[id_var].unique():
            period_of_presence[idx] = df[df[id_var] == idx][
                self.time_series_var
            ].nunique()

        return period_of_presence

    # def _filter_on_exact_conditions(
    #     self,
    #     df: pd.DataFrame,
    #     row: pd.Series(dtype="float64"),
    #     exact_matching: list,
    #     time_series=True,
    # ) -> pd.DataFrame:
    #     """_filter_on_exact_conditions

    #     Matching can be done under constraints : a control individual can only be matched with a treated individual that has the same given characteristics.
    #     To process so, this function filters the dataframe on the exact conditions given by the user.

    #     Args:
    #         df (pd.DataFrame): DataFrame to filter
    #         exact_matching (list): List of var names that need to be the same for the treated and control individual to be matched.
    #         row (pd.Series, optional): row
    #         time_series (bool, optional): _description_. Defaults to True.

    #     Returns:
    #         pd.DataFrame: _description_
    #     """
    #     exact_conditions = (df[exact_matching] == row[exact_matching]).all(axis=1)
    #     return df[exact_conditions]

    # result of the logit, result summary

    def match(
        self, df: pd.DataFrame, exact_matching: Union[List, None] = None
    ) -> pd.DataFrame:
        """lorem ipsum

        Args:
            var (type):

        Return:

        """

        assert df[self.var_treatment].nunique() == 2

        if self.time_series_var != "":
            df = self._write_treatment_period(df)

        # result = self._fit_logit(df, var_logit)        # compute the logit
        # df["propension_score"] = result.predict(df)    #add logit score to dataframe
        # self._check_common_support(df)

        df.reset_index().rename(
            columns={"index": "id_matching"}
        )  # ensure index is reseted before matching
        if self.cylinder:
            period_of_presence = self._identify_period_of_presence(df)
        # match
        self._matching_indices = {}
        for idx1, row in tqdm(
            df.iterrows(), desc="Processing to matching row by row", total=df.shape[0]
        ):
            if row[self.var_treatment] == 1:
                tmp_df_to_match_on = df[df["treated_over_time"] == 0]

                if self.cylinder:
                    tmp_df_to_match_on = tmp_df_to_match_on[
                        tmp_df_to_match_on[self.id_var].map(period_of_presence)
                        == period_of_presence[row[self.id_var]]
                    ]
                if exact_matching:
                    tmp_df_to_match_on = tmp_df_to_match_on[
                        (tmp_df_to_match_on[exact_matching] == row[exact_matching]).all(
                            axis=1
                        )
                    ]

                closest = self._get_closest_neighbors(
                    tmp_df_to_match_on,
                    valeur_propensity=row["propensity_score"],
                    n_neighbors=self.ratio,
                    random_state=self.random_state,
                )

                if closest:
                    self._matching_indices[row[self.id_var]] = (
                        df.loc[closest, self.id_var].unique().tolist()
                    )  # get id_var at index closest

        # add matched index to dataframe
        df["matched_id"] = df[self.id_var].map(self._matching_indices)
        list_counterfactual = df["matched_id"].explode().unique().tolist()

        # filter df to keep only matched rows and treated rows
        df.loc[
            df[self.id_var].isin(
                list_counterfactual + list(self._matching_indices.keys())
            ),
            "to_keep_after_matching",
        ] = 1
        df["to_keep_after_matching"].fillna(0, inplace=True)
        return df

    def get_dataframe_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """The match function returns a dataframe enriched with the matched_id. It identifies the individuals that are part of the treatment group and the control group.
        This function will return a dataframe that is ready for analysis. More precisely, it will return a dataframe that contains the treated individuals and the matched ones duplicated as many times as they have been matched.

        args :
            df (pd.DataFrame : dataframe that has been matched.

        Returns:
            pd.DataFrame : _description_
        """

        assert "matched_id" in df.columns
        assert "to_keep_after_matching" in df.columns

        treated_list = df[df[self.var_treatment] == 1][self.id_var].unique()

        df_for_analysis = []

        for treated in tqdm(treated_list, desc="Adding treated and its controls", total = len(treated_list)):

            tmp_data_treated = df[(df[self.id_var] == treated) & (df["to_keep_after_matching"]==1)]
            period_treatment = (
                tmp_data_treated[[self.time_series_var, "treatment_period"]]
                .set_index(self.time_series_var)
                .to_dict()["treatment_period"]
            )
            df_for_analysis.append(tmp_data_treated)

            for control in tmp_data_treated["matched_id"].explode().unique().tolist():
                tmp_data_control = df[df[self.id_var] == control]
                tmp_data_control.loc[:,"treatment_period"] = tmp_data_control.loc[:,
                    self.time_series_var
                ].map(period_treatment)
                df_for_analysis.append(tmp_data_control)
        
        df_for_analysis = pd.concat(df_for_analysis, axis=0)

        return df_for_analysis

    def _mean_difference_(self, df: pd.DataFrame, var_to_plot: list):
        """_mean_difference_ _summary_

        Args:
            df (pd.DataFrame): _description_
            var_to_plot (list): _description_

        Returns:
            _type_: _description_
        """

        assert "to_keep_after_matching" in df.columns
        assert "treatment" in df.columns

        tmp_left = df.groupby("treatment")[var_to_plot].mean().T
        tmp_left = pd.DataFrame(
            100
            * (abs(tmp_left[1] - tmp_left[0]) / abs(tmp_left[1])).rename(
                "gap between mean of treated group and others before matching"
            )
        )
        tmp_right = (
            df[df["to_keep_after_matching"] == 1]
            .groupby("treatment")[var_to_plot]
            .mean()
            .T
        )
        tmp_right = pd.DataFrame(
            100
            * (abs(tmp_right[1] - tmp_right[0]) / abs(tmp_right[1])).rename(
                "gap between mean of treated group and others after matching"
            )
        )
        tmp = pd.concat([tmp_left, tmp_right], axis=1)
        return tmp
