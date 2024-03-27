"""
Conventions de naming : 

- les noms des classes ne contiennent pas d'underscore, les mots sont séparés par un passage minuscule / majuscule
- un underscore devant un nom de méthode ou de classe indique que ce sont des objets qui ne seront pas appelés par un utilisateur final
- les noms de méthodes peuvent contenir des underscores, et ne contiennent généralement pas de majuscule
"""


from dataclasses import dataclass
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import random 


import seaborn as sns
from abc import abstractmethod, ABC

import statsmodels.api as sm 
import statsmodels.formula.api as smf

from tqdm import tqdm


"""Les deux classes suivantes sont équivalentes"""
class MyClass():

    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

@dataclass
class MyClass():
    arg1 : int
    arg2: str


""" une classe abstraite qui donne un formalisme pour tous les matchers"""
class Matcher(ABC):
    def __init__(self, ratio: int, var_treatment: str):
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
    df = sklearn.datasets.load_iris()
    psm = PropensityScoreMatcher(**kwargs)
    psm.match()
    psm.report()

    ```
    """

    def __init__(self, ratio: int, var_treatment: str, caliper=0.15, random_state=42):
        
        super().__init__(ratio = ratio, var_treatment=var_treatment)
        self.caliper = caliper
        assert (self.caliper >= 0) & (self.caliper <= 1)
        self.random_state = random_state

        pass

    

    def fit_logit_on_df(self, df: pd.DataFrame, var_logit : list) -> pd.DataFrame:
        model = smf.logit(self.var_treatment + " ~ " + f" + ".join([var for var in var_logit]), data = df)
        result = model.fit()
        df["propensity_score"] = result.predict(df)
        print(result.summary()) 
        print("Propensity scores were added to the dataframe in column 'propensity_score'")
        return df
    

    def check_plot_common_support(self, df) : 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.histplot(df, x = "propensity_score", hue = self.var_treatment, ax = ax, bins = 1000, alpha = 0.7, kde =True)
        plt.xlabel('Propensity score')
        plt.ylabel('Density')
        plt.show()
        pass

    def result_summary(self):

        pass



    ## Appliquée en cas réel, la fonction suivante model = smf.logit("Traité ~   lag_r006 + lag_lag_r006 + lag_i255 + lag_lag_i255+ lag_redi_r003 + lag_lag_redi_r003 + I(lag_redi_r003 ** 2)  + lag_redi_r005 +lag_lag_redi_r005+ lag_redi_r101 + lag_lag_redi_r101 +lag_redi_r310 + lag_lag_redi_r310+ lag_redi_r002 + lag_lag_redi_r002  +  is_prefecture  + grand_groupe_français + groupe_etranger + groupe_francofrançais + categorie_entreprise_ETI + categorie_entreprise_PME + categorie_entreprise_MICRO + construction + commerce_auto + sante + administratif + juridique + Hebergement + Immobilier + dom_com_R + dom_com_GA + dom_com_GY + transport +denrees  + services +manufacture + bois", data = df_model_restricted)
#result = model.fit()
#print(result.summary())
#crée une fonction pour la classe qui renvoie le résultat du logit

    # def _fit_logit(self, df: pd.DataFrame, var_logit : list):
    #     model = smf.logit(self.var_treatment + " ~ " + f" + ".join([var for var in var_logit]), data = df)
    #     result = model.fit()
    #     return result
    
    def _get_closest_neighbors(self, df : pd.DataFrame, valeur_propensity : float, n_neighbors:int, random_state : int) -> list :
        diff = (df["propensity_score"]-valeur_propensity).abs()
        random.seed(random_state)
        keep_opt = random.choice(['first','last'])
        index_closest = diff.nsmallest(n_neighbors, keep = keep_opt).index.to_list()
        index_closest_within_caliper = [idx for idx in index_closest if abs(df["propensity_score"][idx] - valeur_propensity) <= self.caliper] 
        return index_closest_within_caliper
    
    def _filter_on_exact_conditions(self, df: pd.DataFrame, row : pd.Series(dtype='float64'),  exact_matching : list) -> pd.DataFrame:
        exact_conditions = (df[exact_matching] == row[exact_matching]).all(axis = 1)
        return df[exact_conditions]
    #result of the logit, result summary 

    def match(self, df: pd.DataFrame,  exact_matching : list) -> pd.DataFrame:
        """lorem ipsum

        Args:
            var (type): 

        Return:

        """
 
        assert df[self.var_treatment].nunique() == 2
  
        # result = self._fit_logit(df, var_logit)        # compute the logit
        # df["propension_score"] = result.predict(df)    #add logit score to dataframe
        #self._check_common_support(df)
        df.reset_index().rename(columns={"index": "id_matching"}) #ensure index is reseted before matching
        # match
        self._matching_indices = {}
        for idx1, row in tqdm(df.iterrows(), desc = "Processing to matching row by row", total = df.shape[0]):
            if row[self.var_treatment] == 1:
                if exact_matching : 
                    closest = self._get_closest_neighbors(self._filter_on_exact_conditions(df, row, exact_matching),valeur_propensity =  row["propensity_score"], n_neighbors = self.ratio +1, random_state = self.random_state)
                else :
                    closest = self._get_closest_neighbors(df,valeur_propensity= row["propensity_score"], n_neighbors = self.ratio + 1 , random_state = self.random_state)
                closest.remove(idx1)
                self._matching_indices[idx1] = closest


        #add matched index to dataframe
        df["matched_index"] = df.index.map(self._matching_indices)
        list_counterfactual = [item for sublist in list(self._matching_indices.values()) for item in sublist]

        #filter df to keep only matched rows and treated rows
        df.loc[df.index.isin(list_counterfactual + list(self._matching_indices.keys())),  "to_keep_after_matching"] = 1
        df["to_keep_after_matching"].fillna(0, inplace = True)


        return df
    
    def _mean_difference_(self, df: pd.DataFrame, var_to_plot : list):
        assert "to_keep_after_matching" in df.columns
        assert "treatment" in df.columns

        tmp_left = df.groupby("treatment")[var_to_plot].mean().T
        tmp_left = pd.DataFrame(100*(abs(tmp_left[1]-tmp_left[0])/abs(tmp_left[1])).rename("gap between mean of treated group and others before matching"))
        tmp_right = df[df["to_keep_after_matching"] == 1].groupby("treatment")[var_to_plot].mean().T
        tmp_right = pd.DataFrame(100*(abs(tmp_right[1]-tmp_right[0])/abs(tmp_right[1])).rename("gap between mean of treated group and others after matching"))
        tmp = pd.concat([tmp_left, tmp_right], axis = 1)
        return tmp 

    def _plot_mean_difference_(self, df: pd.DataFrame, var_to_plot : list):
        tmp_2 = self._mean_difference_(df, var_to_plot)
        tmp_2.plot(kind='bar', figsize=(10, 6))
        formatter = FuncFormatter(lambda x, _: '{:.0%}'.format(x / 100))
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.ylabel('Percentage (%)')
        plt.title('mean difference between treated and control before and after matching')
        plt.show()
        pass

    def report(self, df: pd.DataFrame, var_to_plot : list):
        self._plot_mean_difference_(df, var_to_plot)
        pass




# cette commande renvoie la docstring de la classe PropensityScoreMatcher


# def get_closest_neighbor(serie :pd.Series(), valeur : float, n:int, random_state) -> int : 
#     diff = (serie-valeur).abs()
#     random.seed(random_state)
#     keep_opt = random.choice(['first','last'])
#     index_closest = diff.nsmallest(n, keep = keep_opt).index.to_list()
#     return index_closest