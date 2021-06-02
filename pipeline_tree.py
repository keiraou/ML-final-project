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
CATGORICAL_FEATURES = ['province', 'if_urban',
                 'wealth_index',
                 'if_employment','employment_pay_method','if_earn_more', 
                 'sex_head_household', 'sexual_activity', 'partner_ideal_child', 'money_decide_person']
TARGET_LST = ['if_emo_vio', 'if_phy_vio', 'if_sex_vio', 'if_vio', 'num_vio']


# Read Data
def read_data(csv):
    return pd.read_csv(csv)


# Data Preprocessing
def preprocess_data(df, features_col, target_col, categorical_col):
    df.dropna(subset=[target_col],inplace=True)
    df = fill_categorical_na_vals(df)
    features = df[features_col]
    features = pd.get_dummies(features, columns=categorical_col)
    target = df[target_col]
    return features, target


def impute_missing_median(X_train, X_test, col_lst):
    '''
    Impute missing values of continuous variables using the median value
    '''
    for col in col_lst:
        X_train.loc[(X_train[col] == "don't know") | (X_train[col] == "non-numeric response") , col] = None
        X_test.loc[(X_test[col] == "don't know") | (X_test[col] == "non-numeric response") , col] = None
        median = X_train[col].median()
        X_train[col].fillna(median,inplace=True)
        X_test[col].fillna(median,inplace=True)
    return X_train, X_test


def fill_categorical_na_vals(df):
    '''
    Find colums and rows with missing values. Print rows, returns list of
    columns.
    '''
    df = df.fillna(0)
    return df


# Data Split
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size=0.20, 
                                                    random_state=505)
    X_train, X_test = impute_missing_median(
        X_train, X_test, NUMERIC_FEATURES)
    return X_train, X_test, y_train, y_test


# Train Decision Tree Model
def train_decision_tree(X_train, X_test, y_train, y_test):
    params = {'criterion': ['gini', 'entropy'],
                'max_depth': [3,5,10,15],
                'min_samples_split': [2,5,10]}
    grid_model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=505), 
                              param_grid=params, 
                              cv=10,
                              return_train_score=True,
                              scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                              refit='f1')

    grid_model.fit(X_train, y_train)

    grid_result = pd.DataFrame(grid_model.cv_results_)
    grid_result = grid_result[['params','mean_train_f1','mean_train_accuracy', 'mean_train_precision','mean_train_recall','mean_train_roc_auc']]
    grid_result = grid_result.sort_values(by=['mean_train_f1'], ascending=False)
    return grid_result


# Train Random Forest Model
def train_random_forest(X_train, X_test, y_train, y_test):
    params = {'n_estimators':[10, 100],
              'criterion': ['gini', 'entropy'],
              'max_depth': [3,5,10,15],
              'min_samples_split': [2,5,10]}
    grid_model = GridSearchCV(estimator=RandomForestClassifier(random_state=505), 
                              param_grid=params, 
                              cv=10,
                              return_train_score=True,
                              scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                              refit='f1')

    grid_model.fit(X_train, y_train)

    grid_result = pd.DataFrame(grid_model.cv_results_)
    grid_result = grid_result[['params','mean_train_f1','mean_train_accuracy', 'mean_train_precision','mean_train_recall','mean_train_roc_auc']]
    grid_result = grid_result.sort_values(by=['mean_train_f1'], ascending=False)
    return grid_result


# Evaluate Testing Scores
def evaluate_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plot_precision_recall_curve(model, X_test, y_test)
    results_dict = {}
    results_dict['f1'] = metrics.f1_score(y_test, y_pred)
    results_dict['accuracy'] = metrics.accuracy_score(y_test, y_pred)
    results_dict['precision'] = metrics.precision_score(y_test, y_pred)
    results_dict['recall'] = metrics.recall_score(y_test, y_pred)
    results_dict['roc_auc'] = metrics.roc_auc_score(y_test, y_pred)
    plot_precision_recall_curve(model,X_test,y_test)
    return results_dict


# Feature Importance
def plot_importances(model, X_train, n=10, title=''):
    '''
    Compute the relative importance of selected features in
    the model
    
    Inputs:
    - model
    - n (int): top n features, opt
    - title (str)
    '''
    plt.close()
    importances = model.feature_importances_
    np_features = np.array(X_train.columns)
    sorted_idx = np.argsort(importances)[len(np_features)-n:]
    padding = np.arange(len(sorted_idx)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, np_features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title(title)
    pl.show()