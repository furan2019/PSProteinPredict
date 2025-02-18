import numpy as np
import pandas as pd
import math
import sys
import time
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle, resample
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score

Nops_list = pd.read_csv("nops_list.csv",header=None)
Ps_list = pd.read_csv("speci8_list.csv",header=None)

###################################################################### for TPR value
#Pas = []
#Pas_all = pd.DataFrame()

#for i in range(0, len(Ps_list)):
#    Pas.append(pd.read_csv(str((Ps_list.iloc[i]).values[0]), index_col=0))

#for i in Pas:
#    Pas_all = pd.concat([Pas_all, i])
    
#
#Pas_all_filled = Pas_all.fillna(0)

lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')
pu_model = BaggingClassifierPU(base_estimator=lgb_model)

Ps_list_speci = ['At','Br','Cr','Os','Pp','Sl','Ta','Zm']
fold1_list = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']

#accuracy_lgb_re10 = pd.DataFrame(data=None)
#accuracy_pu_re10 = pd.DataFrame(data=None)

#for re10 in range(0, 10):
#    Nops = []
#    Nops_all = pd.DataFrame()

#    for nn in range(0, len(Ps_list)):
#        Nops.append((pd.read_csv(str((Nops_list.iloc[nn]).values[0]), index_col=0)).sample(n=len(Pas[nn]), random_state=re10))

#    for nnn in Nops:
#        Nops_all = pd.concat([Nops_all, nnn])
        
#    Nops_all_filled = Nops_all.fillna(0)

#    y_pred_pos_acc_lgb = []
#    y_pred_pos_acc_pu = []
    
#    for i in range(0, len(Ps_list)):
#        test_df_pos, train_df_pos, test_df_neg, train_df_neg = [], [], [], []

#        test_df_pos_ = Pas[i]
#        test_df_pos = test_df_pos_.fillna(0)
#        train_df_pos = Pas_all_filled.drop(labels=test_df_pos.index)
        
#        test_df_neg_ = Nops[i]
#        test_df_neg = test_df_neg_.fillna(0)
#        train_df_neg = Nops_all_filled.drop(labels=test_df_neg.index)

#        training_df = pd.concat([train_df_pos, train_df_neg])
#        x_train = np.array(training_df.drop("Score", axis=1))
#        y_train = np.array(training_df.Score)

        #testing_df = pd.concat([test_df_pos,test_df_neg])
        #x_test = np.array(testing_df.drop("Score", axis=1))
        #y_test = np.array(testing_df.Score)
#        x_test_pos = np.array(test_df_pos.drop("Score", axis=1))
#        y_test_pos = np.array(test_df_pos.Score)
    
#        lgb_model.fit(x_train, y_train)
#        y_pred_pos_lgb = lgb_model.predict(x_test_pos)
#        pos_acc_lgb = accuracy_score(y_test_pos, y_pred_pos_lgb)
    
#        pu_model.fit(x_train, y_train)
#        y_pred_pos_pu = pu_model.predict(x_test_pos)
#        pos_acc_pu = accuracy_score(y_test_pos, y_pred_pos_pu)
       
#        y_pred_pos_acc_lgb.append(pos_acc_lgb)
#        y_pred_pos_acc_pu.append(pos_acc_pu)
            
#    accuracy_lgb_re10[re10] = y_pred_pos_acc_lgb
#    accuracy_pu_re10[re10] = y_pred_pos_acc_pu

#print(accuracy_lgb_re10)
#print(accuracy_pu_re10)
      
#accuracy_lgb_re10.columns = fold1_list
#accuracy_lgb_re10.index = Ps_list_speci
#accuracy_lgb_re10.to_csv("lgb_interspecies_TPR202412.csv")
  
#accuracy_pu_re10.columns = fold1_list
#accuracy_pu_re10.index = Ps_list_speci
#accuracy_pu_re10.to_csv("pu_interspecies_TPR202412.csv")
  
############################################################################# for AUC plot
Nops = []
Pas = []
Nops_all = pd.DataFrame()
Pas_all = pd.DataFrame()

for i in range(0,len(Ps_list)):
    Pas.append(pd.read_csv(str((Ps_list.iloc[i]).values[0]),index_col=0))
    Nops.append((pd.read_csv(str((Nops_list.iloc[i]).values[0]),index_col=0)).sample(n=len(Pas[i]), random_state=202501))

for i in Nops:
    Nops_all = pd.concat([Nops_all,i])

for i in Pas:
    Pas_all = pd.concat([Pas_all, i])


Pas_all_filled = Pas_all.fillna(0)  
Nops_all_filled = Nops_all.fillna(0)

plt.figure()

for i in range(0,len(Ps_list)):
    test_df_pos_ = Pas[i]
    test_df_pos = test_df_pos_.fillna(0)
    train_df_pos = Pas_all_filled.drop(labels=test_df_pos.index)
        
    test_df_neg_ = Nops[i]
    test_df_neg = test_df_neg_.fillna(0)
    train_df_neg = Nops_all_filled.drop(labels=test_df_neg.index)

    training_df = pd.concat([train_df_pos,train_df_neg])
    x_train = np.array(training_df.drop("Score", axis=1))
    y_train = np.array(training_df.Score)

    testing_df = pd.concat([test_df_pos,test_df_neg])
    x_test = np.array(testing_df.drop("Score", axis=1))
    y_test = np.array(testing_df.Score)
    
    pu_model.fit(x_train, y_train)
    fpr_pu_lgb, tpr_pu_lgb, _ = roc_curve(y_test, pu_model.predict_proba(x_test)[:, 1])
    roc_auc_pu_lgb = auc(fpr_pu_lgb, tpr_pu_lgb)
    plt.plot(fpr_pu_lgb, tpr_pu_lgb, label=f'{Ps_list_speci[i]} , {roc_auc_pu_lgb:.2f}', linestyle='--')
    
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show() 
    
