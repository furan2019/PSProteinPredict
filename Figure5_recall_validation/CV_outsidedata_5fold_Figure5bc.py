import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
 
# ps data set
positive_samples_08 = pd.read_csv("unlabel_2_ps.csv", index_col=0)
# nonps data set
negative_samples_01 = pd.read_csv("unlabel_2_nops.csv", index_col=0)

#validata set     
positive_valid = pd.read_csv("ps_outside_feature.csv", index_col=0)

negative_valid = pd.read_csv("nops_outside_feature.csv", index_col=0)
negative_valid_or = negative_valid

#model
lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')
PU_lgb = BaggingClassifierPU(
    lgb_model,
    n_estimators=100,
    max_samples=100,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1)
        
#evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

#
lgb_f1,pu_f1 = [],[]
#roc_auc_pu_lgb,roc_auc_lgb = [],[]
# ROC plot
plt.figure()

for re10 in range(0, 5):
	positive_samples_or = positive_samples_08.sample(n=5000, random_state=re10)
	positive_samples = positive_samples_or.fillna(0)
	
	negative_samples_or = negative_samples_01.sample(n=5000, random_state=re10)
	negative_samples = negative_samples_or.fillna(0)
	
	#training data
	train_data = pd.concat([positive_samples, negative_samples])
	X_train = train_data.drop('Score', axis=1).values
	y_train = train_data['Score'].values

	#validation data
	positive_valid_fill = positive_valid.fillna(0)
	negative_valid_fill = negative_valid_or.fillna(0)
	
	
	valid_data = pd.concat([positive_valid_fill, negative_valid_fill])
	X_test = valid_data.drop('Score', axis=1).values
	y_test = valid_data['Score'].values

	X_test_pos = positive_valid_fill.drop('Score', axis=1).values
	y_test_pos = positive_valid_fill['Score'].values

	X_test_neg = negative_valid_fill.drop('Score', axis=1).values
	y_test_neg = negative_valid_fill['Score'].values
	
	lgbm_f1score = evaluate_model(lgb_model, X_train, y_train, X_test, y_test)
	pu_f1score = evaluate_model(PU_lgb, X_train, y_train, X_test, y_test)
    
	lgb_f1.append(lgbm_f1score)
	pu_f1.append(pu_f1score)
	
	fpr_pu_lgb, tpr_pu_lgb, _ = roc_curve(y_test, PU_lgb.predict_proba(X_test)[:, 1])
	roc_auc_pu_lgb = auc(fpr_pu_lgb, tpr_pu_lgb)

	fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_model.predict_proba(X_test)[:, 1])
	roc_auc_lgb = auc(fpr_lgb, tpr_lgb)
	
	plt.plot(fpr_pu_lgb, tpr_pu_lgb, label=f'PU_LGB = {roc_auc_pu_lgb:.2f}')
	plt.plot(fpr_lgb, tpr_lgb, label=f'LGB = {roc_auc_lgb:.2f}')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()

predict_outside = pd.DataFrame({
                "LGB": lgb_f1,
                "PU_lgb": pu_f1,
            })
             
predict_outside.to_csv("predict_outside_re10.csv")

