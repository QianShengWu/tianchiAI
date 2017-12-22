# -*- coding: utf-8 -*-
# @Date    : 2017/12/19
# @Author  : wqs
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

train = pd.read_excel('./data/train.xlsx')
test = pd.read_excel('./data/testA.xlsx')

train = train.fillna(0)
test = test.fillna(0)

# le = LabelEncoder()
# le.fit(train['TOOL_ID'].values)
# train['TOOL_ID'] = le.transform(train['TOOL_ID'])
# test['TOOL_ID'] = le.transform(test['TOOL_ID'])

feature_columns = [x for x in train.columns if x not in ['ID', 'Y'] and train[x].dtype != object]
X_train, y = train[feature_columns], train['Y']
X_test = test[feature_columns]

# rf = RandomForestRegressor(n_estimators=200, n_jobs=3)
# rf.fit(X_train, y)
# # print(rf.feature_importances_)
#
# model = SelectFromModel(rf, prefit=True)
# X_train_new, X_test_new = model.transform(X_train), model.transform(X_test)
# rf.fit(X_train_new, y)
# y_predict = rf.predict(X_test_new)

corr = {}
for f in X_train.columns:
    data = X_train[f]
    corr[f] = pearsonr(data.values, y.values)[0]
feature = []
for k, v in corr.items():
    if abs(v) >= 0.1:
        feature.append(k)

X_train = train[feature_columns]
X_test = test[feature_columns]

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                         colsample_bytree=1, max_depth=7)
model.fit(X_train, y)
y_predict = model.predict(X_test)

with open('./result/wqs_{}.csv'.format(time.strftime("%Y-%m-%d-%H-%M-%S")), 'w') as f:
    for id, y in zip(test['ID'], y_predict):
        f.write('{},{}\n'.format(id, y))
