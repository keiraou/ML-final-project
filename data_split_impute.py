import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

FEATURES = ['province','age','education', 'if_urban',
                 'wealth_index', 'if_own_house',
                 'if_employment', 'if_employment_current','employment_pay_method', 'if_earn_more',
                 'partner_edu', 
                 'num_child','sex_head_household', 'sexual_activity', 'ideal_num_child', 'partner_ideal_child', 'money_decide_person']
NUMERIC_FEATURES = ['age','education','if_own_house','if_employment_current','partner_edu','num_child','ideal_num_child']
CATGORICAL_FEATURES = ['if_urban',
                 'wealth_index',
                 'employment_pay_method','if_earn_more', 
                 'sex_head_household', 'sexual_activity', 'partner_ideal_child', 'money_decide_person']
TARGET_LST = ['if_emo_vio', 'if_phy_vio', 'if_sex_vio', 'if_vio', 'num_vio']

#need hot-code
need_one_hot=['if_urban','wealth_index','if_earn_more','sex_head_household', \
              'partner_ideal_child','money_decide_person','country']
#already dummy
dummy=['if_own_house','if_employment_current']
#need normalize
need_normalize=['age','education','num_household','num_child','partner_edu']
features_col = need_normalize + dummy + need_one_hot



# Data Split
def split_data(features, target, test_size=0.20, random_state=505):
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size, 
                                                    random_state)
    return X_train, X_test, y_train, y_test


def detect_null(df):
    return df.isnull.any()


def detect_outliers(df, feature):
    '''
    Detect possible outliers in a dataframe.
    Inputs: 
        df: dataframe
        feature: str, the feature to be detected on finding outliers.
    Return: a list of possible outlier values.
    '''

    mean = df[feature].mean()
    std = df[feature].std()
    outliers = []
    for index, data in df.iterrows():
        if abs((data.loc[feature] - mean)/std)> 2:
            outliers.append(index)
    return outliers


def impute_missing_median(df, col_lst):
    '''
    Impute missing values of continuous variables using the median value
    '''
    for col in col_lst:
        df.loc[(df[col] == "don't know") | (df[col] == "non-numeric response") , col] = None
        median = df[col].median()
        df[col].fillna(median,inplace=True)
    return df




