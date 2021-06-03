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
from imblearn.ensemble import BalancedRandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# FEATURES = ['province','age','education', 'if_urban',
#                  'wealth_index', 'if_own_house',
#                  'if_employment', 'if_employment_current','employment_pay_method', 'if_earn_more',
#                  'partner_edu', 
#                  'num_child','sex_head_household', 'sexual_activity', 'ideal_num_child', 'partner_ideal_child', 'money_decide_person']
# NUMERIC_FEATURES = ['age','education','if_own_house','if_employment_current','partner_edu','num_child','ideal_num_child']
# CATGORICAL_FEATURES = ['province', 'if_urban',
#                  'wealth_index',
#                  'if_employment','employment_pay_method','if_earn_more', 
#                  'sex_head_household', 'sexual_activity', 'partner_ideal_child', 'money_decide_person']
# TARGET_LST = ['if_emo_vio', 'if_phy_vio', 'if_sex_vio', 'if_vio', 'num_vio']
# RESULT_MODEL = pd.DataFrame(columns = ('country','target','model','params','f1','accuracy','precision','recall','roc_auc','model_object'))


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
        X_train.loc[(X_train[col] == "don't know") | (X_train[col] == "non-numeric response") | (X_train[col] == "up to god"), col] = None
        X_test.loc[(X_test[col] == "don't know") | (X_test[col] == "non-numeric response") | (X_test[col] == "up to god") , col] = None
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
def split_data(features, target, NUMERIC_FEATURES):
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size=0.20, 
                                                    random_state=505)
    X_train, X_test = impute_missing_median(
        X_train, X_test, NUMERIC_FEATURES)
    return X_train, X_test, y_train, y_test


# Train Decision Tree Model
def train_decision_tree(X_train, X_test, y_train, y_test, country, target_col):
    params = {'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10,15],
                'min_samples_split': [2,5]}
    grid_model = GridSearchCV(estimator=DecisionTreeClassifier(), 
                              param_grid=params, 
                              cv=10,
                              return_train_score=True,
                              scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                              refit='f1')

    grid_model.fit(X_train, y_train)
    grid_result = pd.DataFrame(grid_model.cv_results_)
    grid_result = grid_result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
    grid_result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
    grid_result['country'] = country
    grid_result['target'] = target_col
    grid_result['model'] = 'DecisionTree'
    grid_result = grid_result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
    return grid_result


# Train Random Forest Model
def train_random_forest(X_train, X_test, y_train, y_test, country, target_col):
    params = {'n_estimators':[10],
              'criterion': ['gini', 'entropy'],
              'max_depth': [5, 10, 15],
              'min_samples_split': [2,5]}
    grid_model = GridSearchCV(
        estimator=RandomForestClassifier(), 
                              param_grid=params, 
                              cv=10,
                              return_train_score=True,
                              scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                              refit='recall')

    grid_model.fit(X_train, y_train)
    grid_result = pd.DataFrame(grid_model.cv_results_)
    grid_result = grid_result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
    grid_result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
    grid_result['country'] = country
    grid_result['target'] = target_col
    grid_result['model'] = 'RandomForest'
    grid_result = grid_result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
    return grid_result


# Train Gradient Boosting Classifier
def train_gradient_boosting(X_train, X_test, y_train, y_test, country, target_col):
    params = {'n_estimators':[10],
              'loss': ['deviance'],
              'criterion': ['friedman_mse', 'mae'],
              'max_depth': [5,10,15],
              'min_samples_split': [2,5]}
    grid_model = GridSearchCV(
        estimator=GradientBoostingClassifier(), 
                              param_grid=params, 
                              cv=10,
                              return_train_score=True,
                              scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                              refit='recall')

    grid_model.fit(X_train, y_train)
    grid_result = pd.DataFrame(grid_model.cv_results_)
    grid_result = grid_result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
    grid_result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
    grid_result['country'] = country
    grid_result['target'] = target_col
    grid_result['model'] = 'GradientBoosting'
    grid_result = grid_result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
    return grid_result


# Evaluate Testing Scores
def evaluate_test(model, X_test, y_test, row):
    y_pred = model.predict(X_test)
    plot_precision_recall_curve(model, X_test, y_test)
    # results_dict = {}
    row['f1'] = metrics.f1_score(y_test, y_pred)
    row['accuracy'] = metrics.accuracy_score(y_test, y_pred)
    row['precision'] = metrics.precision_score(y_test, y_pred)
    row['recall'] = metrics.recall_score(y_test, y_pred)
    row['roc_auc'] = metrics.roc_auc_score(y_test, y_pred)
    row['model_object'] = model
    plot_precision_recall_curve(model,X_test,y_test)
    return row


# Feature Importance
def plot_importances(model, X_train, n=10, title=''):
    '''
    Compute the relative importance of selected features in
    the model
    
    Inputs:
    - model
    - X_train
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


def one_country_model_decision_tree(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL):
    '''
    Use decision tree to model data of one country
    '''
    print("\n \n Country: ", country, "\n\n")
    df_lst = []
    for target_col in TARGET_LST:
        print("\n \n Target: ", target_col, "\n\n")
        features, target = preprocess_data(df, FEATURES, target_col, CATGORICAL_FEATURES)
        X_train, X_test, y_train, y_test = split_data(features, target, NUMERIC_FEATURES)
        grid_result= train_decision_tree(X_train, X_test, y_train, y_test, country, target_col)
        # for params in grid_result['params']:
        #     row_dict = {}
        #     row_dict['country'] = country
        #     row_dict['target'] = target_col
        #     row_dict['model'] = 'DecisionTree'
        #     row_dict['params'] = params
        #     print('Params: ', params)
        #     good_model = DecisionTreeClassifier(**params).fit(X_train, y_train)
        #     row = evaluate_test(good_model, X_test, y_test, row_dict)
        #     print(row)
        #     # plot_importances(good_model, X_train, n=10, title= target_col + str(params))
        #     RESULT_MODEL = RESULT_MODEL.append(row, ignore_index=True)
        df_lst.append(grid_result)
    concated_grid_result = pd.concat(df_lst)
    return concated_grid_result

def one_country_model_random_forest(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL):
    '''
    Use random forest to model data of one country
    '''
    df_lst = []
    for target_col in TARGET_LST:
        print("\n \n Target: ", target_col, "\n\n")
        features, target = preprocess_data(df, FEATURES, target_col, CATGORICAL_FEATURES)
        X_train, X_test, y_train, y_test = split_data(features, target, NUMERIC_FEATURES)
        grid_result= train_random_forest(X_train, X_test, y_train, y_test, country, target_col)
        # for params in grid_result['params']:
        #     row_dict = {}
        #     row_dict['country'] = country
        #     row_dict['target'] = target_col
        #     row_dict['model'] = 'RandomForest'
        #     row_dict['params'] = params
        #     # print('Params: ', params)
        #     good_model = RandomForestClassifier(**params).fit(X_train, y_train)
        #     row = evaluate_test(good_model, X_test, y_test, row_dict)
        #     # print(row)
        #     # plot_importances(good_model, X_train, n=10, title= target_col + str(params))
        #     RESULT_MODEL = RESULT_MODEL.append(row, ignore_index=True)
        df_lst.append(grid_result)
    concated_grid_result = pd.concat(df_lst)
    return concated_grid_result


def one_country_model_gradient_boosting(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL):
    '''
    Use gradient boosting to model data of one country
    '''
    df_lst = []
    for target_col in TARGET_LST:
        print("\n \n Target: ", target_col, "\n\n")
        features, target = preprocess_data(df, FEATURES, target_col, CATGORICAL_FEATURES)
        X_train, X_test, y_train, y_test = split_data(features, target, NUMERIC_FEATURES)
        grid_result= train_gradient_boosting(X_train, X_test, y_train, y_test, country, target_col)
        # for params in grid_result['params']:
        #     row_dict = {}
        #     row_dict['country'] = country
        #     row_dict['target'] = target_col
        #     row_dict['model'] = 'GradientBoosting'
        #     row_dict['params'] = params
        #     # print('Params: ', params)
        #     good_model = GradientBoostingClassifier(**params).fit(X_train, y_train)
        #     row = evaluate_test(good_model, X_test, y_test, row_dict)
        #     # print(row)
        #     # plot_importances(good_model, X_train, n=10, title= target_col + str(params))
        #     RESULT_MODEL = RESULT_MODEL.append(row, ignore_index=True)
        df_lst.append(grid_result)
    concated_grid_result = pd.concat(df_lst)
    return concated_grid_result

def one_country_model_all_tree(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL):
    '''
    Use decision tree, random forest, gradient boosting
    to model data of one country
    '''
    df_brf = one_country_model_balanced_rf(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL)
    df_wrf = one_country_model_weighted_rf(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL)
    df_dt = one_country_model_decision_tree(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL)
    df_rf = one_country_model_random_forest(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL)
    # df_gb = one_country_model_gradient_boosting(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL)
    result = pd.concat([df_brf, df_wrf, df_dt, df_rf])
    # result = pd.concat([df_brf, df_wrf, df_dt, df_rf, df_gb])
    return result


def one_country_model_balanced_rf(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL):
    '''
    Use Balanced Random Forest to model data of one country
    '''
    df_lst = []
    for target_col in TARGET_LST:
        print("\n \n Target: ", target_col, "\n\n")
        features, target = preprocess_data(df, FEATURES, target_col, CATGORICAL_FEATURES)
        X_train, X_test, y_train, y_test = split_data(features, target, NUMERIC_FEATURES)
        params = {'criterion': ['gini', 'entropy'],
                    'max_depth': [5,10,15],
                    'min_samples_split': [2,5]}
        grid_model = GridSearchCV(estimator=BalancedRandomForestClassifier(), 
                                  param_grid=params, 
                                  cv=10,
                                  return_train_score=True,
                                  scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                                  refit='f1')

        grid_model.fit(X_train, y_train)
        grid_result = pd.DataFrame(grid_model.cv_results_)
        grid_result = grid_result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
        grid_result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
        grid_result['country'] = country
        grid_result['target'] = target_col
        grid_result['model'] = 'BalancedRandomForest'
        grid_result = grid_result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
        df_lst.append(grid_result)
    concated_grid_result = pd.concat(df_lst)
    return concated_grid_result


def one_country_model_weighted_rf(df, country, TARGET_LST, FEATURES, CATGORICAL_FEATURES, NUMERIC_FEATURES, RESULT_MODEL):
    '''
    Use Weighted Random Forest to model data of one country
    '''
    df_lst = []
    for target_col in TARGET_LST:
        print("\n \n Target: ", target_col, "\n\n")
        features, target = preprocess_data(df, FEATURES, target_col, CATGORICAL_FEATURES)
        X_train, X_test, y_train, y_test = split_data(features, target, NUMERIC_FEATURES)

        params = {'n_estimators':[10],
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10,15],
                'class_weight':['balanced', 'balanced_subsample'],
                'min_samples_split': [2,5]}
        grid_model = GridSearchCV(
            estimator=RandomForestClassifier(), 
                                param_grid=params, 
                                cv=10,
                                return_train_score=True,
                                scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                                refit='recall')

        grid_model.fit(X_train, y_train)
        grid_result = pd.DataFrame(grid_model.cv_results_)
        grid_result = grid_result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
        grid_result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
        grid_result['country'] = country
        grid_result['target'] = target_col
        grid_result['model'] = 'WeightedRandomForest'
        grid_result = grid_result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
        df_lst.append(grid_result)
    concated_grid_result = pd.concat(df_lst)
    return concated_grid_result

