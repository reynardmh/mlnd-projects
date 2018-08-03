import StringIO
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cross_validation as cv
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# potentiall interesting features
col_desc = {
    'C150_4_POOLED': 'Completion 4yr pooled',
    'C150_L4_POOLED': 'Completion <4yr pooled',
    'CCSIZSET': 'Carnegie classification-Size & settings',
    'CCUGPROF': 'Carnegie classification-Undergrad profile ',
    'CCBASIC': 'Carnegie classification-basic',
    'CONTROL': 'Control (public/private)',
    'RET_FT4': 'Retention 4yr',
    'RET_FTL4': 'Retention <4yr',
    'ACTCMMID': 'ACT',
    'SAT_AVG': 'SAT',
    'SAT_AVG_ALL': 'SAT all',
    'SATVRMID': 'SAT reading',
    'SATMTMID': 'SAT math',
    'SATWRMID': 'SAT writing',
    'AVGFACSAL': 'Avg faculty salary',
    'PFTFAC': 'Full time faculty rate',
    'ADM_RATE_ALL': 'Admission rate',
    'DISTANCEONLY': 'Distance only',
    'NPT4_PUB': 'Avg net price title IV institut public',
    'NPT4_PRIV': 'Avg net price title IV institut private',
    'NUM4_PUB': 'Num Title IV student, public',
    'NUM4_PRIV': 'Num Title IV student, private',
    'COSTT4_A': 'Avg cost academic year',
    'COSTT4_P': 'Avg cost program year',
    'TUITIONFEE_IN': 'In state tuition',
    'TUITIONFEE_OUT': 'Out of state tuition',
    'TUITIONFEE_PROG': 'Tuition fee program year',
    'TUITFTE': 'Net revenue per FTE student',
    'INEXPFTE': 'Expense per FTE student',
    'PCTPELL': '% Pell Grant receiver',
    'PCTFLOAN': '% Fed student loan',
    'UG25abv': '% undergrad > 25 yr',
    'PFTFTUG1_EF': 'Undergrad 1st-time degree seeking',
    'UGDS': 'Number of Undergrad degree seeking',
    'PAR_ED_PCT_1STGEN': '% 1st gen students',
    'PAR_ED_PCT_MS': '% parent education middle school',
    'PAR_ED_PCT_HS': '% parent education high school',
    'PAR_ED_PCT_PS': '% parent education post secondary',
    'DEP_INC_AVG': 'Avg income dependent stu',
    'IND_INC_AVG': 'Avg income independent stu',
    'DEBT_MDN': 'Median debt',
    'GRAD_DEBT_MDN': 'Median debt complete',
    'WDRAW_DEBT_MDN': 'Median debt non-completer',

    'LOCALE': 'Locale',
    'region': 'region',
    'PREDDEG': 'Predominant degree',

    # added cols
    'L4_COLLEGE': '<4 years college',
    'C150': 'Completion',
    'RET_FT': 'Retention',
    'NPT4': 'Avg net price Title IV',
    'NUM4': 'Num Title IV student',
}

# convert 1 dimensional N array into 2 dimensional Nx1
def reshape_y(y):
    return y.reshape((len(y), 1))

def plot_feature_vs_completion(cols, data_for_plotting, xscale=None, categorical=False):
    data = data_for_plotting['data']
    data_L4 = data_for_plotting['data_L4']
    data_4 = data_for_plotting['data_4']
    data_control1 = data_for_plotting['data_control1']
    data_control2 = data_for_plotting['data_control2']
    data_control3 = data_for_plotting['data_control3']

    for col in cols:
        if col not in data.columns:
            continue
        print "--- {} ---".format(col_desc[col])
        if categorical:
            plt.ylabel("Count")
            plt.xlabel(col_desc[col])
            plt.hist(data[col].dropna())
            plt.show()
        else:
            print data[col].describe()
            plt.boxplot([data[col].dropna()], labels=[col_desc[col]])
            plt.show()

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,4))
        psize = 2
        alpha = 0.8
        ax1.scatter(data_4[col], data_4['C150'], c='b', s=psize, linewidths=0)
        ax1.scatter(data_L4[col], data_L4['C150'], c='r', s=psize, linewidths=0, alpha=alpha)
        ax1.set_title("Blue:4yr, Red:<4yr")
        ax1.set_xlabel(col_desc[col])
        ax1.set_ylabel("Completion")
        if xscale:
            ax1.set_xscale(xscale)

        ax2.scatter(data_control1[col], data_control1['C150'], c='b', s=psize, linewidths=0)
        ax2.scatter(data_control2[col], data_control2['C150'], c='r', s=psize, linewidths=0, alpha=alpha)
        ax2.scatter(data_control3[col], data_control3['C150'], c='g', s=psize, linewidths=0, alpha=alpha)
        ax2.set_title("Blue:public, Red:private nonprofit, Green:private profit")
        ax2.set_xlabel(col_desc[col])
        ax2.set_ylabel("Completion")
        if xscale:
            ax2.set_xscale(xscale)

        plt.show()

def plot_feature_vs_retention(cols, data_for_plotting, xscale=None):
    data = data_for_plotting['data']
    data_L4 = data_for_plotting['data_L4']
    data_4 = data_for_plotting['data_4']
    data_control1 = data_for_plotting['data_control1']
    data_control2 = data_for_plotting['data_control2']
    data_control3 = data_for_plotting['data_control3']

    for col in cols:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,4))
        psize = 2
        alpha = 0.8
        ax1.scatter(data_4[col], data_4['RET_FT'], c='b', s=psize, linewidths=0)
        ax1.scatter(data_L4[col], data_L4['RET_FT'], c='r', s=psize, linewidths=0, alpha=alpha)
        ax1.set_title("Blue:4yr, Red:<4yr")
        ax1.set_xlabel(col_desc[col])
        ax1.set_ylabel("Retention")
        if xscale:
            ax1.set_xscale(xscale)

        ax2.scatter(data_control1[col], data_control1['RET_FT'], c='b', s=psize, linewidths=0)
        ax2.scatter(data_control2[col], data_control2['RET_FT'], c='r', s=psize, linewidths=0, alpha=alpha)
        ax2.scatter(data_control3[col], data_control3['RET_FT'], c='g', s=psize, linewidths=0, alpha=alpha)
        ax2.set_title("Blue:public, Red:private nonprofit, Green:private profit")
        ax2.set_xlabel(col_desc[col])
        ax2.set_ylabel("Retention")
        if xscale:
            ax2.set_xscale(xscale)

        plt.show()

# merge some columns that are mutually exclusive (C150_4_POOLED & C150_L4_POOLED)
def preprocess_data(orig_data):
    added_cols = ['L4_COLLEGE', 'C150', 'RET_FT', 'NPT4', 'NUM4']
    selected_keys = list(set(col_desc.keys()) - set(added_cols))
    data = orig_data[sorted(selected_keys)].copy()

    # add column that indicates whether it's a less than 4yr college
    data['L4_COLLEGE'] = data.C150_4_POOLED.isnull()

    # combine completion data for 4 year and <4 year institution
    data['C150'] = pd.concat([data.C150_4_POOLED.dropna(), data.C150_L4_POOLED.dropna()]).reindex_like(data)

    # combine retention data for 4 year and <4 year institution
    data['RET_FT'] = pd.concat([data.RET_FT4.dropna(), data.RET_FTL4.dropna()]).reindex_like(data)

    # combine net price title iv for public and private
    data['NPT4'] = pd.concat([data.NPT4_PRIV.dropna(), data.NPT4_PUB.dropna()]).reindex_like(data)
    data['NUM4'] = pd.concat([data.NUM4_PRIV.dropna(), data.NUM4_PUB.dropna()]).reindex_like(data)

    # columns to clean up after combining
    del_columns = ['NPT4_PUB', 'NPT4_PRIV', 'NUM4_PUB', 'NUM4_PRIV', 'C150_4_POOLED', 'C150_L4_POOLED']
    for col in del_columns:
        if col in data.keys():
            del data[col]
            # del col_desc[col]

    data = data[~data['C150'].isnull()]
    data = data[~data['RET_FT'].isnull()]

    # remove data containing 'PrivacySuppressed'
    for col in data.columns:
        if data.dtypes[col] == 'object':
            data[col] = data[col].replace(['PrivacySuppressed'], [float('NaN')]).astype(float)

    return data

def print_num_data_for_each_features(data):
    print "Number of available data for each feature (not counting the NaN values)"
    tmp = (data.isnull().sum() - len(data)) * -1
    for k, v in tmp.iteritems():
        print "{0:20s}{1:45s}{2:5d}".format(k, col_desc[k], v)

def print_r2score(reg, X, y, test=False):
    t = 'test ' if test else 'train'
    r2score = metrics.r2_score(y, reg.predict(np.array(X)))
    print "R2 score on {} data: {}".format(t, r2score)
    return r2score

def print_mse(reg, X, y, test=False):
    t = 'test ' if test else 'train'
    mse = metrics.mean_squared_error(y, reg.predict(np.array(X)))
    print "MSE on {} data: {}".format(t, mse)
    return mse

def print_mae(reg, X, y, test=False):
    t = 'test ' if test else 'train'
    mae = metrics.mean_absolute_error(y, reg.predict(np.array(X)))
    print "MAE on {} data: {}".format(t, mae)
    return mae

r2_scorer = metrics.make_scorer(metrics.r2_score, greater_is_better=True)
mse_scorer = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)
mae_scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=False)
default_scorer = mae_scorer

def split_y(y_train, y_test):
    return y_train[:,0], y_test[:,0], y_train[:,1], y_test[:,1]

def print_r2_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test):
    print "--- R2 Completion ---"
    if hasattr(reg1, 'best_params_'):
        print "best params: {}".format(reg1.best_params_)
    r2_train_reg1 = print_r2score(reg1, X_train, y1_train)
    r2_test_reg1  = print_r2score(reg1, X_test, y1_test, test=True)

    print "--- R2 Retention ---"
    if hasattr(reg2, 'best_params_'):
        print "best params: {}".format(reg2.best_params_)
    r2_train_reg2 = print_r2score(reg2, X_train, y2_train)
    r2_test_reg2  = print_r2score(reg2, X_test, y2_test, test=True)

    return {'reg1': {'train': r2_train_reg1, 'test': r2_test_reg1 },
            'reg2': {'train': r2_train_reg2, 'test': r2_test_reg2 }}

def print_mse_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test):
    print "--- MSE Completion ---"
    mse_train_reg1 = print_mse(reg1, X_train, y1_train)
    mse_test_reg1 = print_mse(reg1, X_test, y1_test, test=True)
    print "--- MSE Retention ---"
    mse_train_reg2 = print_mse(reg2, X_train, y2_train)
    mse_test_reg2 = print_mse(reg2, X_test, y2_test, test=True)
    return {'reg1': {'train': mse_train_reg1, 'test': mse_test_reg1 },
            'reg2': {'train': mse_train_reg2, 'test': mse_test_reg2 }}

def print_mae_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test):
    print "--- MAE Completion ---"
    mae_train_reg1 = print_mae(reg1, X_train, y1_train)
    mae_test_reg1 = print_mae(reg1, X_test, y1_test, test=True)
    print "--- MAE Retention ---"
    mae_train_reg2 = print_mae(reg2, X_train, y2_train)
    mae_test_reg2 = print_mae(reg2, X_test, y2_test, test=True)
    return {'reg1': {'train': mae_train_reg1, 'test': mae_test_reg1 },
            'reg2': {'train': mae_train_reg2, 'test': mae_test_reg2 }}

def plot_err_metric(err, metric=None, model=None):
    plt.title('{} - {}'.format(metric, model))
    plt.bar(np.array(range(2)), [err['reg1']['train'], err['reg2']['train']], width=0.4, color='r')
    plt.bar(np.array(range(2))+0.4, [err['reg1']['test'], err['reg2']['test']], width=0.4, color='b')
    plt.ylabel(metric)
    plt.legend(['Train', 'Test'])
    plt.xticks(np.array(range(4))+ 0.4, ['Completion', 'Retention'])
    plt.show()

def build_SVR_model(X_train, X_test, y_train, y_test, cv=3, params=None, scorer=None):
    if scorer == None:
        scorer = default_scorer
    y1_train, y1_test, y2_train, y2_test = split_y(y_train, y_test)

    if params == None:
        params = {'C': np.logspace(-1, 1, 2), 'gamma': np.logspace(-1, 1, 2), 'epsilon': np.logspace(-1, 1, 2)}
    reg = SVR()
    best_reg1 = GridSearchCV(reg, params, scoring=scorer, cv=cv)
    best_reg1.fit(X_train, y1_train)

    reg = SVR()
    best_reg2 = GridSearchCV(reg, params, scoring=scorer, cv=cv)
    best_reg2.fit(X_train, y2_train)

    r2 = print_r2_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mse = print_mse_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mae = print_mae_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    plot_err_metric(r2, metric='R2 score', model='SVR')
    plot_err_metric(mse, metric='MSE', model='SVR')
    plot_err_metric(mae, metric='MAE', model='SVR')

    return {'reg1': best_reg1, 'reg2': best_reg2, 'r2': r2, 'mse': mse, 'mae': mae}

def build_DecisionTree_model(X_train, X_test, y_train, y_test, cv=3, scorer=None):
    if scorer == None:
        scorer = default_scorer
    y1_train, y1_test, y2_train, y2_test = split_y(y_train, y_test)

    parameters = {'max_depth': range(1,10)} # , 'min_samples_leaf': [4,5,6,7]}
    reg = DecisionTreeRegressor()
    best_reg1 = GridSearchCV(reg, parameters, scoring=scorer, cv=cv)
    best_reg1.fit(X_train, y1_train)

    reg = DecisionTreeRegressor()
    best_reg2 = GridSearchCV(reg, parameters, scoring=scorer, cv=cv)
    best_reg2.fit(X_train, y2_train)

    r2 = print_r2_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mse = print_mse_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mae = print_mae_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    plot_err_metric(r2, metric='R2 score', model='Decision Tree')
    plot_err_metric(mse, metric='MSE', model='Decision Tree')
    plot_err_metric(mae, metric='MAE', model='Decision Tree')

    return {'reg1': best_reg1, 'reg2': best_reg2, 'r2': r2, 'mse': mse, 'mae': mae}

def build_KNN_model(X_train, X_test, y_train, y_test, cv=3, scorer=None):
    if scorer == None:
        scorer = default_scorer
    y1_train, y1_test, y2_train, y2_test = split_y(y_train, y_test)

    parameters = {'n_neighbors': range(5,20)}
    reg = KNeighborsRegressor()
    best_reg1 = GridSearchCV(reg, parameters, scoring=scorer, cv=cv)
    best_reg1.fit(X_train, y1_train)

    parameters = {'n_neighbors': range(5, 20)}
    reg = KNeighborsRegressor()
    best_reg2 = GridSearchCV(reg, parameters, scoring=scorer, cv=cv)
    best_reg2.fit(X_train, y2_train)

    r2 = print_r2_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mse = print_mse_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mae = print_mae_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    plot_err_metric(r2, metric='R2 score', model='KNN')
    plot_err_metric(mse, metric='MSE', model='KNN')
    plot_err_metric(mae, metric='MAE', model='KNN')

    return {'reg1': best_reg1, 'reg2': best_reg2, 'r2': r2, 'mse': mse, 'mae': mae}

def build_RandomForest_model(X_train, X_test, y_train, y_test, n_estimators=10, scorer=None):
    if scorer == None:
        scorer = default_scorer
    y1_train, y1_test, y2_train, y2_test = split_y(y_train, y_test)

    reg1 = RandomForestRegressor(n_estimators=n_estimators)
    reg1.fit(X_train, y1_train)

    reg2 = RandomForestRegressor(n_estimators=n_estimators)
    reg2.fit(X_train, y2_train)

    r2 = print_r2_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mse = print_mse_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    mae = print_mae_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test)
    plot_err_metric(r2, metric='R2 score', model='Random Forest')
    plot_err_metric(mse, metric='MSE', model='Random Forest')
    plot_err_metric(mae, metric='MAE', model='Random Forest')

    return {'reg1': reg1, 'reg2': reg2, 'r2': r2, 'mse': mse, 'mae': mae}

def plot_compare_metric(models, model_names=[], metric=None):
    train_scores1, test_scores1, train_scores2, test_scores2 = [], [], [], []
    for m in models:
        train_scores1.append(m[metric]['reg1']['train'])
        test_scores1.append(m[metric]['reg1']['test'])
        train_scores2.append(m[metric]['reg2']['train'])
        test_scores2.append(m[metric]['reg2']['test'])

    plt.title('Completion models - {}'.format(metric.upper()))
    plt.bar(np.array(range(len(models))), train_scores1, width=0.4, color='r')
    plt.bar(np.array(range(len(models)))+0.4, test_scores1, width=0.4, color='b')
    plt.ylabel(metric.upper())
    plt.legend(['Train', 'Test'])
    plt.xticks(np.array(range(len(models)+2))+ 0.4, model_names)
    plt.show()

    plt.title('Retention models - {}'.format(metric.upper()))
    plt.bar(np.array(range(len(models))), train_scores2, width=0.4, color='r')
    plt.bar(np.array(range(len(models)))+0.4, test_scores2, width=0.4, color='b')
    plt.ylabel(metric.upper())
    plt.legend(['Train', 'Test'])
    plt.xticks(np.array(range(len(models)+2))+ 0.4, model_names)
    plt.show()


def plot_model_improvement(models1, models2, model_names=[], metric=None):
    completion_scores1, completion_scores2 = [], []
    retention_scores1, retention_scores2 = [], []
    for m in models1:
        completion_scores1.append(m[metric]['reg1']['test'])
        retention_scores1.append(m[metric]['reg2']['test'])
    for m in models2:
        completion_scores2.append(m[metric]['reg1']['test'])
        retention_scores2.append(m[metric]['reg2']['test'])

    plt.title('Completion {}: 19 vs 83 features'.format(metric.upper()))
    plt.bar(np.array(range(len(models1))), completion_scores1, width=0.4, color='#aaccff')
    plt.bar(np.array(range(len(models1)))+0.4, completion_scores2, width=0.4, color='#88aacc')
    plt.ylabel(metric.upper())
    plt.legend(['19', '83'])
    plt.xticks(np.array(range(len(models1)+2))+ 0.4, model_names)
    plt.show()

    plt.title('Retention {}: 19 vs 83 features'.format(metric.upper()))
    plt.bar(np.array(range(len(models1))), retention_scores1, width=0.4, color='#aacc77')
    plt.bar(np.array(range(len(models1)))+0.4, retention_scores2, width=0.4, color='#88aa55')
    plt.ylabel(metric.upper())
    plt.legend(['19', '83'])
    plt.xticks(np.array(range(len(models1)+2))+ 0.4, model_names)
    plt.show()

# Histogram plot of error prediction for a model
def hist_plot_delta(model, X, y, model_name=None, nested=False):
    bucket = np.linspace(0, 1, 20)
    y1, y2 = y[:,0], y[:,1]

    y1_predicted = model['reg1'].predict(X)
    if nested:
        X_tmp = np.concatenate((X, reshape_y(y1_predicted)), axis=1)
        y2_predicted = model['reg2'].predict(X_tmp)
    else:
        y2_predicted = model['reg2'].predict(X)
    delta1 = abs(y1 - y1_predicted)
    delta2 = abs(y2 - y2_predicted)

    plt.title('Prediction error distribution, Completion {}'.format(model_name))
    plt.xlabel('prediction error')
    plt.ylabel('count')
    plt.hist(delta1, bucket)
    plt.show()

    plt.title('Prediction error distribution, Retention {}'.format(model_name))
    plt.xlabel('prediction error')
    plt.ylabel('count')
    plt.hist(delta2, bucket)
    plt.show()

# Comparison of histogram plot of 2 models
def compare_hist_plot_delta(m1, m2, X, y, model_names=None, nested=[False, False]):
    bucket = np.linspace(0, 1, 20)
    y1, y2 = y[:,0], y[:,1]

    y1_predicted, y2_predicted = {}, {}
    y1_predicted['m1'] = m1['reg1'].predict(X)
    y1_predicted['m2'] = m2['reg1'].predict(X)
    if nested[0]:
        X_tmp = np.concatenate((X, reshape_y(y1_predicted['m1'])), axis=1)
        y2_predicted['m1'] = m1['reg2'].predict(X_tmp)
    else:
        y2_predicted['m1'] = m1['reg2'].predict(X)
    if nested[1]:
        X_tmp = np.concatenate((X, reshape_y(y1_predicted['m2'])), axis=1)
        y2_predicted['m2'] = m2['reg2'].predict(X_tmp)
    else:
        y2_predicted['m2'] = m2['reg2'].predict(X)

    delta1 = { 'm1': abs(y1 - y1_predicted['m1']),
               'm2': abs(y1 - y1_predicted['m2'])}
    delta2 = { 'm1': abs(y2 - y2_predicted['m1']),
               'm2': abs(y2 - y2_predicted['m2'])}

    plt.title('Prediction error distribution - Completion models')
    plt.xlabel('prediction error')
    plt.ylabel('count')
    plt.hist([delta1['m1'], delta1['m2']], bucket, alpha=0.5)
    plt.legend(model_names)
    plt.show()

    plt.title('Prediction error distribution - Retention models')
    plt.xlabel('prediction error')
    plt.ylabel('count')
    plt.hist([delta2['m1'], delta2['m2']], bucket, alpha=0.5)
    plt.legend(model_names)
    plt.show()
