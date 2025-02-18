import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score
from lightgbm import LGBMClassifier

Nops_list = pd.read_csv("nops_list.csv",header=None)
Ps_list = pd.read_csv("speci8_list.csv",header=None)

Nops = []
Pas = []
Nops_all = pd.DataFrame()
Pas_all = pd.DataFrame()

for i in range(0,len(Ps_list)):
    Pas.append(pd.read_csv(str((Ps_list.iloc[i]).values[0]),index_col=0))
    Nops.append((pd.read_csv(str((Nops_list.iloc[i]).values[0]),index_col=0)).sample(n=len(Pas[i]), random_state=2025))

for i in Nops:
    Nops_all = pd.concat([Nops_all,i])

for i in Pas:
    Pas_all = pd.concat([Pas_all, i])

Pas_all_filled = Pas_all.fillna(0) 
Nops_all_filled = Nops_all.fillna(0)
y_pred_pos_acc = pd.DataFrame()
lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')

plt.figure()

for i in range(0,7):
    test_df_pos = Pas_all_filled.sample(n=800, random_state=2025)
    train_df_pos = Pas_all_filled.drop(labels=test_df_pos.index)
    test_df_neg = Nops_all_filled.sample(n=800, random_state=2025)
    train_df_neg = Nops_all_filled.drop(labels=test_df_neg.index)

    test_df_pos_f1 = test_df_pos.iloc[:, [i, 7]]
    train_df_pos_f1 = train_df_pos.iloc[:, [i, 7]]
    test_df_neg_f1 = test_df_neg.iloc[:, [i, 7]]
    train_df_neg_f1 = train_df_neg.iloc[:, [i, 7]]
    
    training_df = pd.concat([train_df_pos_f1,train_df_neg_f1])
    x_train = np.array(training_df.drop("Score", axis=1))
    y_train = np.array(training_df.Score)

    testing_df = pd.concat([test_df_pos_f1,test_df_neg_f1])
    x_test = np.array(testing_df.drop("Score", axis=1))
    y_test = np.array(testing_df.Score)
    x_test_pos = np.array(test_df_pos_f1.drop("Score", axis=1))
    y_test_pos = np.array(test_df_pos_f1.Score)
    
    lgb_model.fit(x_train, y_train)
    fpr_pu_lgb, tpr_pu_lgb, _ = roc_curve(y_test, lgb_model.predict_proba(x_test)[:, 1])
    roc_auc_pu_lgb = auc(fpr_pu_lgb, tpr_pu_lgb)
    plt.plot(fpr_pu_lgb, tpr_pu_lgb, label=f'{training_df.columns[0]} , {roc_auc_pu_lgb:.2f}', linestyle='--')

    y_pred_pos = lgb_model.predict(x_test_pos)
    y_pred_pos_acc[i] = [accuracy_score(y_test_pos, y_pred_pos)]

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

y_pred_pos_acc.columns = Pas_all.columns[:7]
y_pred_pos_acc.to_csv("Lgb_individualfeature_bulk_code1_TPR2025.csv")

