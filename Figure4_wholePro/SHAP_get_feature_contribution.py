import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score
import matplotlib.pyplot as plt

# ps data set
ps_samples = pd.read_csv("Pas_all.csv", index_col=0)
positive_samples_new = pd.read_csv("unlabel_2_ps.csv", index_col=0)
positive_samples_or = positive_samples_new.sample(n=20000, random_state=4048)
positive_samples_all = pd.concat([ps_samples, positive_samples_or])
#
positive_samples = positive_samples_all.fillna(0)
n_positive = len(positive_samples)

# nonps data set
negative_samples_new = pd.read_csv("unlabel_2_nops.csv", index_col=0)
negative_samples_or = negative_samples_new.sample(n=n_positive, random_state=4048)
#
negative_samples = negative_samples_or.fillna(0)

# training data set
train_data = pd.concat([positive_samples, negative_samples])
X = train_data.drop('Score', axis=1).values
y = train_data['Score'].values

#test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')

######################## for plot1 plot2
lgb_model.fit(X_train, y_train)
explainer = shap.TreeExplainer(lgb_model)

###plot1
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1],X_test,plot_type="dot")

###plot2
#shap_interaction_values = explainer.shap_interaction_values(X_test)
#shap.summary_plot(shap_interaction_values,X_test)

##################### for plot3
#PU_lgb = BaggingClassifierPU(lgb_model)
#PU_lgb.fit(X_train, y_train)
#explainer = shap.KernelExplainer(PU_lgb.predict_proba, X_train.sample(n=100))
#shap_values = explainer.shap_values(X_test.sample(n=5))
#shap.summary_plot(shap_values,X_test.sample(n=5),plot_type="dot")


