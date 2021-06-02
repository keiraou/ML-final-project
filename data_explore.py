
import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

## Explore the dataframe

def generate_corr_graph(df):
    '''
    Generate correlation graph based on given dataframe to
    show correlations between variables
    '''
    return df.corr(method='pearson').style.background_gradient(cmap='Blues')


def summ_categorical_col(df, col):
    '''
    Summarize categorical values based on given dataframe and column
    '''
    sns.set(rc={'figure.figsize':(20, 15)})
    df[col].hist()
    return df[col].value_counts()


def summ_continuous_col(df, col):
    '''
    Summarize continuous values based on given dataframe and column
    '''
    sns.set(rc={'figure.figsize':(20, 15)})
    df[col].hist()
    return df[col].describe()


def multi_scatter(df, vars_interest):
    '''
    Plots a scatter for each pair of selected variables

    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of strings): list of column names
    '''
    df_subs = df[vars_interest]
    pd.plotting.scatter_matrix(df_subs, alpha = 0.7, figsize = (14,8), diagonal = 'kde')


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