import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, roc_auc_score,plot_precision_recall_curve
import datetime
import warnings
warnings.filterwarnings('ignore')


#check na
def plot_df_na(df):

    return sns.heatmap(df.isnull(), cbar=False)


def check_na_proportion(df):

    val_dic = {}
    for col in df:
        val_prop = df[col].isna().astype(int).sum()/df.shape[0]
        val_dic[col] = val_prop
    return pd.DataFrame(val_dic)


def drop_na(df, subset):

    for col in df.columns:  
        df[col][df[col] == "don't know"] = None
        df[col][df[col] == "non-numeric response"] = None
        
    return df.dropna(subset=subset, inplace=True)


def preprocess_data(df, features_col, target_col, need_one_hot):

    #create features df
    features = df[features_col]
    
    #one-hot encoding
    features = pd.get_dummies(features, columns=need_one_hot)
    
    #create target
    target = df[target_col]
    
    return features, target
    

# Data Split
def split_data(features, target, need_normalize):
    '''
    features, target: dataframe
    '''
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size=0.20, 
                                                    random_state=0)
    for col in need_normalize:
        X_train[col] = pd.to_numeric(X_train[col])
        X_test[col] = pd.to_numeric(X_test[col])
        X_train[col] = (X_train[col]-X_train[col].mean())/X_train[col].std()
        X_test[col] = (X_test[col]-X_train[col].mean())/X_train[col].std()
    
    return  X_train, X_test, y_train, y_test      


def impute_missing_median(X_train, X_test):

    for col in X_train.columns:  
        median = X_train[col].median()
        X_train[col].fillna(median, inplace=True)
        X_test[col].fillna(median, inplace=True)
    
    return X_train, X_test


def train_logistic_regression_SMOTE(X_train, y_train, country, target_col):
    pipeline = make_pipeline(SMOTE(random_state=0), LogisticRegression(random_state=0))

    params = {'logisticregression__penalty': ['l2'],
            'logisticregression__C': [0.01, 0.1, 1, 10, 100],
            'logisticregression__solver': ['lbfgs']}

    model = GridSearchCV(estimator=pipeline,
                        param_grid=params, 
                        cv=10,
                        return_train_score=True,
                        scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                        refit='f1')

    model.fit(X_train, y_train)
    result = pd.DataFrame(model.cv_results_)
    result = result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
    result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
    result['country'] = country
    result['target'] = target_col
    result['model'] = 'LogisticRegression'
    result = result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
    return result, model.best_estimator_.get_params()['steps'][1][1]


def train_GaussianNB_SMOTE(X_train, y_train, country, target_col):
    
    pipeline = make_pipeline((SMOTE(random_state=0)), GaussianNB())
    params = {}
    model = GridSearchCV(estimator=pipeline,
                        param_grid=params,
                        cv=10,
                        return_train_score=True,
                        scoring= ['f1', 'accuracy','precision','recall','roc_auc'],
                        refit = 'accuracy')
    
    model.fit(X_train, y_train)
    result = pd.DataFrame(model.cv_results_)
    result = result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
    result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
    result['country'] = country
    result['target'] = target_col
    result['model'] = 'GaussianNB'
    result = result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
    return result, model.best_estimator_.get_params()['steps'][1][1]


def train_LinearSVC_SMOTE(X_train, y_train, country, target_col):
    pipeline = make_pipeline(SMOTE(random_state=0), LinearSVC(random_state=0))

    params = {'linearsvc__C': [0.01, 0.1, 1, 10, 100]}

    model = GridSearchCV(estimator=pipeline,
                        param_grid=params, 
                        cv=10,
                        return_train_score=True,
                        scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                        refit='f1')

    model.fit(X_train, y_train)
    result = pd.DataFrame(model.cv_results_)
    result = result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
    result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
    result['country'] = country
    result['target'] = target_col
    result['model'] = 'LinearSVC'
    result = result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
    return result, model.best_estimator_.get_params()['steps'][1][1]


def train_logistic_regression(X_train, y_train, country, target_col):
    #create new grid for LogisticRegression
    params =  {"penalty": ['l2'],
                "C": [0.01, 0.1, 1, 10, 100]} 

    model = GridSearchCV(estimator=LogisticRegression(random_state=0),
                    param_grid=params, 
                    cv=10,
                    return_train_score=True,
                    scoring=['f1', 'accuracy','precision','recall','roc_auc'],
                    refit='f1')

    model.fit(X_train, y_train)
    result = pd.DataFrame(model.cv_results_)
    result = result[['params','mean_test_f1','mean_test_accuracy', 'mean_test_precision','mean_test_recall','mean_test_roc_auc']]
    result.columns = ['params','f1','accuracy','precision','recall','roc_auc']
    result['country'] = country
    result['target'] = target_col
    result['model'] = 'LogisticRegression'
    result = result[['country','target','model','params','f1','accuracy','precision','recall','roc_auc']]
    return result, model.best_estimator_
    

def evaluate_test(model, X_test, y_test):
    #calculate y_score for the random forest model
    # pred_prob = model.predict_proba(X_test)[:,1]

    # sort_y = sorted(zip(y_test, pred_prob), key = lambda x: x[1], reverse=True)
    # y_test, pred_prob = zip(*sort_y)
    # y_test = np.array(y_test)
    # y_pred = np.array(pred_prob)
      
    # n = int(len(y_pred)*p)
    # y_pred[: n] = 1
    # y_pred[n: ] = 0

    y_pred = model.predict(X_test)

    row = {}
    row['f1'] = f1_score(y_test, y_pred)
    row['accuracy'] = accuracy_score(y_test, y_pred)
    row['precision'] = precision_score(y_test, y_pred)
    row['recall'] = recall_score(y_test, y_pred)
    row['roc_auc'] = roc_auc_score(y_test, y_pred)
    plot_precision_recall_curve(model,X_test,y_test)
    return row


def get_important_attributes(features_col, best_model):
    coeffs = pd.DataFrame.from_dict({'predictor':features_col,
                                        'coefficient':best_model.coef_.flatten(),
                                        'abs_coeffient':abs(best_model.coef_.flatten())})

    coeffs.sort_values(by='abs_coeffient', inplace=True, ascending=False)
    return coeffs


def plot_top10_attributes(coeffs, best_model):
    plt.figure(figsize=(10,10), dpi=256)
    sns.barplot(x="coefficient", y = "predictor", palette="ch:s=.25,rot=-.25", \
                    data = coeffs[0:10]).set_title('''Top Ten Features for Emotional 
                    Violence Prediction {}'''.format(best_model))
    plt.show()


def prepare_data_contry(df, features_col, target_col, dummy, need_one_hot, need_normalize, all):
    
    if all:
        drop_na(df, target_col + dummy + need_one_hot)
        features, target = preprocess_data(df, features_col, target_col, need_one_hot)

    else:
        drop_na(df, target_col + dummy + need_one_hot[:-1])
        features, target = preprocess_data(df, features_col[:-1], target_col, need_one_hot[:-1])
    
    X_train, X_test, y_train, y_test = split_data(features, target, need_normalize)
    X_train, X_test = impute_missing_median(X_train, X_test)
    return X_train, X_test, y_train, y_test


def name_target(y_train, y_test):
    y_train1 = y_train['if_emo_vio']
    y_train2 = y_train['if_phy_vio']
    y_train3 = y_train['if_sex_vio']
    y_train4 = y_train['if_vio']

    y_test1 = y_test['if_emo_vio']
    y_test2 = y_test['if_phy_vio']
    y_test3 = y_test['if_sex_vio']
    y_test4 = y_test['if_vio']
    return y_train1, y_train2, y_train3, y_train4, y_test1, y_test2, y_test3, y_test4

    
def analyze_country(X_train, X_test, y_train, y_test, country, target_col, classification):
    if classification == 'LR':
        result, best_model = train_logistic_regression_SMOTE(X_train, y_train, country, target_col)  
    if classification == 'SVC':
        result, best_model = train_LinearSVC_SMOTE(X_train, y_train, country, target_col)   
    if classification == 'NB':
        result, best_model = train_GaussianNB_SMOTE(X_train, y_train, country, target_col) 

    eval = evaluate_test(best_model, X_test, y_test)

    if classification == 'NB':
        return result, eval

    coeffs = get_important_attributes(X_train.columns, best_model)
    plot_top10_attributes(coeffs, best_model)
    return result, eval, coeffs

    # if classification != 'SVC':
    #     eval = evaluate_test(best_model, X_test, y_test, p)




    # if classification == 'LR':
    #     return result, eval, coeffs
    
    # if classification == 'SVC':
    #     return result, coeffs




    