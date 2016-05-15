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

    # 'age_entry': 'Average age of entry',
    # 'agege24': 'Percent of students over 23 at entry',
    # 'female': 'Share of female students',
    # 'married': 'Share of married students',
    # 'dependent': 'Share of dependent students',
    # 'veteran': 'Share of veteran students',
    # 'first_gen': 'Share of first-generation students',
    # 'faminc': 'Average family income',
    # 'md_faminc': 'Median family income',
    # 'faminc_ind': 'Average family income for independent students',
    # 'median_hh_inc': 'Median household income',
    # 'poverty_rate': 'Poverty rate',
    # 'unemp_rate': 'Unemployment rate',

    # added cols
    'L4_COLLEGE': '<4 years college',
    'C150': 'Completion',
    'RET_FT': 'Retention',
    'NPT4': 'Avg net price Title IV',
    'NUM4': 'Num Title IV student',
}

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

r2_scorer = metrics.make_scorer(metrics.r2_score, greater_is_better=True)
mse_scorer = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)
mae_scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=False)
default_scorer = mse_scorer

def split_y(y_train, y_test):
    return y_train[:,0], y_test[:,0], y_train[:,1], y_test[:,1]

def print_r2_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test, model=None):
    print "--- Completion ---"
    if hasattr(reg1, 'best_params_'):
        print "best params: {}".format(reg1.best_params_)
    r2score_train_reg1 = print_r2score(reg1, X_train, y1_train)
    r2score_test_reg1  = print_r2score(reg1, X_test, y1_test, test=True)

    print "--- Retention ---"
    if hasattr(reg2, 'best_params_'):
        print "best params: {}".format(reg2.best_params_)
    r2score_train_reg2 = print_r2score(reg2, X_train, y2_train)
    r2score_test_reg2  = print_r2score(reg2, X_test, y2_test, test=True)

    if model != None:
        plt.title(model)
    ax = plt.subplot(111)
    ax.set_ylim([0, 1])
    ax.bar(np.array(range(2)), [r2score_train_reg1, r2score_train_reg2], width=0.4, color='r')
    ax.bar(np.array(range(2))+0.4, [r2score_test_reg1, r2score_test_reg2], width=0.4, color='b')
    plt.ylabel('R2 score')
    plt.legend(['Train', 'Test'])
    plt.xticks(np.array(range(5))+ 0.4, ['Completion', 'Retention'])
    plt.show()

    return (r2score_test_reg1, r2score_test_reg2)

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

    r2score_reg1, r2score_reg2 = print_r2_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test, model='SVR')

    return (best_reg1, best_reg2, r2score_reg1, r2score_reg2)

def build_DecisionTree_model(X_train, X_test, y_train, y_test, cv=3, scorer=None):
    if scorer == None:
        scorer = default_scorer
    y1_train, y1_test, y2_train, y2_test = split_y(y_train, y_test)

    parameters = {'max_depth': range(1,10) } # , 'min_samples_leaf': [4,5,6,7]}
    reg = DecisionTreeRegressor()
    best_reg1 = GridSearchCV(reg, parameters, scoring=scorer, cv=cv)
    best_reg1.fit(X_train, y1_train)

    reg = DecisionTreeRegressor()
    best_reg2 = GridSearchCV(reg, parameters, scoring=scorer, cv=cv)
    best_reg2.fit(X_train, y2_train)

    r2score_reg1, r2score_reg2 = print_r2_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test, model='Decision Tree')

    return (best_reg1, best_reg2, r2score_reg1, r2score_reg2)

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

    r2score_reg1, r2score_reg2 = print_r2_summary(best_reg1, best_reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test, model='KNN')
    return (best_reg1, best_reg2, r2score_reg1, r2score_reg2)

def build_RandomForest_model(X_train, X_test, y_train, y_test, n_estimators=10, scorer=None):
    if scorer == None:
        scorer = default_scorer
    y1_train, y1_test, y2_train, y2_test = split_y(y_train, y_test)

    reg1 = RandomForestRegressor(n_estimators=n_estimators)
    reg1.fit(X_train, y1_train)

    reg2 = RandomForestRegressor(n_estimators=n_estimators)
    reg2.fit(X_train, y2_train)

    r2score_reg1, r2score_reg2 = print_r2_summary(reg1, reg2, X_train, X_test, y1_train, y1_test, y2_train, y2_test, model='Random Forest')

    return (reg1, reg2, r2score_reg1, r2score_reg2)
