X = data[['CONTROL', 'DEBT_MDN', 'DEP_INC_AVG', 'GRAD_DEBT_MDN', 'IND_INC_AVG', 'INEXPFTE', 'PAR_ED_PCT_1STGEN',
          'PAR_ED_PCT_HS', 'PAR_ED_PCT_MS', 'PAR_ED_PCT_PS', 'PCTFLOAN', 'PCTPELL', 'UG25abv', 'UGDS',
          'WDRAW_DEBT_MDN', 'L4_COLLEGE', 'NPT4', 'NUM4',
          'PFTFTUG1_EF', 'PFTFAC',
          'SAT_AVG_ALL', 'ACTCMMID', 'ADM_RATE_ALL', 'AVGFACSAL', 'COSTT4_A',
          'C150', 'RET_FT']]

X = X[~(X.RET_FT == 0)]
X = X[~(X.C150 == 0)]
X = X[~((X.C150 == 1) & (X.RET_FT < 0.8))]
X = X[~((X.RET_FT == 1) & (X.C150 < 0.8))]

y = X[['C150', 'RET_FT']]
X = X.drop('C150', 1)
X = X.drop('RET_FT', 1)
print X.shape
print y.shape


from sklearn.svm import SVR
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def build_SAT_regressor(data):
    dat = data[['PAR_ED_PCT_PS', 'PAR_ED_PCT_HS', 'SAT_AVG_ALL']]
    dat = dat.dropna()

    X_train, X_test, y_train, y_test = cv.train_test_split(dat[['PAR_ED_PCT_PS', 'PAR_ED_PCT_HS']], dat['SAT_AVG_ALL'], train_size=0.8)

#     params = {'C': np.logspace(-1, 1, 2), 'gamma': np.logspace(-1, 1, 2), 'epsilon': np.logspace(-1, 1, 2)}
#     scorer = metrics.make_scorer(metrics.r2_score, greater_is_better=True)
#     reg = SVR()
#     reg = GridSearchCV(reg, params, scoring=scorer, cv=4)
#     reg.fit(X_train, y_train)
#     p5lib.print_r2score(reg, X_test, y_test)

    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(X_train, y_train)
    p5lib.print_r2score(reg, X_test, y_test)
    return reg


def build_ACT_regressor(data):
    dat = data[['PAR_ED_PCT_PS', 'PAR_ED_PCT_HS', 'ACTCMMID']]
    dat = dat.dropna()
#     print dat.shape

    X_train, X_test, y_train, y_test = cv.train_test_split(dat[['PAR_ED_PCT_PS', 'PAR_ED_PCT_HS']], dat['ACTCMMID'], train_size=0.8)

    params = {'C': np.logspace(-1, 1, 2), 'gamma': np.logspace(-1, 1, 2), 'epsilon': np.logspace(-1, 1, 2)}
    scorer = metrics.make_scorer(metrics.r2_score, greater_is_better=True)
    reg = SVR()
    reg = GridSearchCV(reg, params, scoring=scorer, cv=4)
    reg.fit(X_train, y_train)
    p5lib.print_r2score(reg, X_test, y_test)

#     reg = RandomForestRegressor(n_estimators=100)
#     reg.fit(X_train, y_train)
#     p5lib.print_r2score(reg, X_test, y_test)
    return reg


sat_reg = build_SAT_regressor(X)
act_reg = build_ACT_regressor(X)

print len(X)
print X['SAT_AVG_ALL'].isnull().sum()
print X['ACTCMMID'].isnull().sum()
print X['SAT_AVG_ALL'].describe()
print X['ACTCMMID'].describe()

import math
# sat_reg.predict([[0.3, 0.7]])
for i, x in X['SAT_AVG_ALL'].iteritems():
    if math.isnan(x) and (not math.isnan(X['PAR_ED_PCT_PS'][i]) and not math.isnan(X['PAR_ED_PCT_HS'][i])):
        X.loc[i, 'SAT_AVG_ALL'] = sat_reg.predict([[X['PAR_ED_PCT_PS'][i], X['PAR_ED_PCT_HS'][i]]])

for i, x in X['ACTCMMID'].iteritems():
    if math.isnan(x) and (not math.isnan(X['PAR_ED_PCT_PS'][i]) and not math.isnan(X['PAR_ED_PCT_HS'][i])):
        X.loc[i, 'ACTCMMID'] = act_reg.predict([[X['PAR_ED_PCT_PS'][i], X['PAR_ED_PCT_HS'][i]]])

print X['SAT_AVG_ALL'].isnull().sum()
print X['ACTCMMID'].isnull().sum()
print X['SAT_AVG_ALL'].describe()
print X['ACTCMMID'].describe()

fill_cols_with_mean = ['DEBT_MDN', 'DEP_INC_AVG', 'GRAD_DEBT_MDN', 'IND_INC_AVG', 'INEXPFTE', 'WDRAW_DEBT_MDN',
                       'PAR_ED_PCT_1STGEN', 'PAR_ED_PCT_HS', 'PAR_ED_PCT_MS', 'PAR_ED_PCT_PS', 'PCTFLOAN', 'PCTPELL',
                       'UG25abv', 'NPT4', 'NUM4', 'PFTFTUG1_EF', 'PFTFAC', 'SAT_AVG_ALL', 'ACTCMMID', 'ADM_RATE_ALL',
                       'AVGFACSAL', 'COSTT4_A']

for col in fill_cols_with_mean:
    if col in X:
        X[col] = X[col].fillna(X[col].mean())
