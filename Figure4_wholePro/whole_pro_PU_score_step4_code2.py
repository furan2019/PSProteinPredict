### STEP4 ###

import numpy as np
from lightgbm import LGBMClassifier
import pandas as pd
import os
from baggingPU import BaggingClassifierPU
from sklearn.metrics import accuracy_score, f1_score

#psdata set
positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)

background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
background_index = background_samples_.index

#valid_data set
positive_valid = pd.read_csv("positive_test.csv", index_col=0)
background_valid = pd.read_csv("background_test.csv", index_col=0)
background_valid_ = background_valid.sample(n=len(positive_valid), random_state=2024)
data_valid = pd.concat([positive_valid, background_valid_])

#training data set
train_data = pd.concat([positive_samples_, background_samples_])
X = train_data.drop('Score', axis=1).values
y = train_data['Score'].values
train_index = train_data.index

base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', random_state = 2)
cf = BaggingClassifierPU(base_estimator, n_estimators=100, max_samples=100, bootstrap=True, oob_score=True, n_jobs=-1)
cf.fit(X, y)
pu_score = cf.oob_decision_function_[:, 1]

predResults = pd.DataFrame({
    "ProteinIndex": train_index,
    "OOB_score": pu_score
}, columns=["ProteinIndex", "OOB_score"])

background_protein_puscore = predResults.tail(144000)
background_protein_puscore.to_csv("OOB_score.csv",index=False)

