#Cross-validation to obtain the best LGB parameters

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV

# get training set
positive_samples = pd.read_csv("LGB_ps_final08_.csv", index_col=0)
negative_samples = pd.read_csv("LGB_neg_final01_.csv", index_col=0)

train_data = pd.concat([positive_samples, negative_samples])
X = train_data.drop('Score', axis=1).values
y = train_data['Score'].values

# 5-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# evaluation indicators
scoring = {
    'Accuracy' : make_scorer(accuracy_score),
    'AUC' : make_scorer(roc_auc_score)
}
best_params_list = []

# model
#lgb_model = lgb.LGBMClassifier(objective='binary', boosting_type='dart', metric='auc')
lgb_model = lgb.LGBMClassifier(objective='binary', boosting_type='dart', learning_rate=0.5, bagging_fraction=0.5,
                               feature_fraction=0.8, min_child_samples=18, num_leaves=5)

# parameters
param_grid = {
    'num_leaves': [5, 31],
    'learning_rate': [0.01, 0.5],
    'feature_fraction': [0.8, 1.0],
    'bagging_fraction': [0.5, 1.0],
    'min_child_samples': [18, 19, 20]
}

# GridSearchCV
for train_idx, valid_idx in kfold.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    positive_idx = np.where(y_train == 1)[0]
    negative_idx = np.where(y_train == 0)[0]
    train_neg_idx = np.random.choice(negative_idx, size=len(positive_idx), replace=False)

    X_train_selected = np.concatenate((X_train[positive_idx], X_train[train_neg_idx]), axis=0)
    y_train_selected = np.concatenate((y_train[positive_idx], y_train[train_neg_idx]), axis=0)

    grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring=scoring, refit='AUC', cv=3)
    grid_search.fit(X_train_selected, y_train_selected)

    best_params_list.append(grid_search.best_params_)


# output
for i, best_params in enumerate(best_params_list, start=1):
    print(f"Interation {i} - Best parameters:", best_params)
