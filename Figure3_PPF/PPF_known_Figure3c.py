import argparse, pickle, os, joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle, resample
from baggingPU import BaggingClassifierPU
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error, confusion_matrix

def GetProdictPS_lgb(x_train, y_train, x_test, y_test):
    classifier = LGBMClassifier(objective='binary', boosting_type='dart')
    rf = classifier
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)    
    accuracy = accuracy_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)
    f1_s = f1_score(y_pred, y_test)
    return accuracy, recall, f1_s

def GetProdictPS_pu(X_test, X_train, Y_test, Y_train):
    estimator = LGBMClassifier(objective='binary', boosting_type='dart')
    bc = BaggingClassifierPU(base_estimator=estimator)
    bc.fit(X_train, Y_train)
    Y_pred = bc.predict(X_test)   
    accuracy = accuracy_score(Y_pred, Y_test)
    recall = recall_score(Y_pred, Y_test)
    f1_s = f1_score(Y_pred, Y_test)
    return accuracy, recall, f1_s
    
Nops_df_org = pd.read_csv("NonPS_7features+PPF.csv", index_col=0)
Ps_df_org = pd.read_csv("PS_7features+PPF.csv", index_col=0)

mean_values_ps = Ps_df_org.mean()
Pas_all_filled = Ps_df_org.fillna(mean_values_ps)
mean_values_no = Nops_df_org.mean()
Nops_all_filled = Nops_df_org.fillna(mean_values_no)

accuracy_f8_lgb_re10, accuracy_f8_pu_re10, accuracy_f1_lgb_re10,  accuracy_f7_lgb_re10= (
    pd.DataFrame(data=None), pd.DataFrame(data=None), pd.DataFrame(data=None), pd.DataFrame(data=None))
f1_f8_lgb_re10, f1_f8_pu_re10, f1_f1_lgb_re10,  f1_f7_lgb_re10= (
    pd.DataFrame(data=None), pd.DataFrame(data=None), pd.DataFrame(data=None), pd.DataFrame(data=None))

#num = 200
num = 100

for re10 in range(0, 10):

    Nops_df_re = Nops_all_filled.sample(n=len(Pas_all_filled), random_state=re10)
    
    accuracy_f8_lgb, accuracy_f8_pu, accuracy_f1_lgb, accuracy_f7_lgb = [], [], [], []
    f1_f8_lgb, f1_f8_pu, f1_f1_lgb, f1_f7_lgb = [], [], [], []

    for i in range(0,10):

        ###for lgb f8
        test_df_pos = Pas_all_filled.sample(n=num, random_state=i)
        train_df_pos = Pas_all_filled.drop(labels=test_df_pos.index)
        test_df_neg = Nops_df_re.sample(n=num, random_state=i)
        train_df_neg = Nops_df_re.drop(labels=test_df_neg.index)

        training_df = pd.concat([train_df_pos, train_df_neg])
        X_train = np.array(training_df.drop("Score", axis=1))
        Y_train = np.array(training_df.Score)

        x_test_pos = np.array(test_df_pos.drop("Score", axis=1))
        y_test_pos = np.array(test_df_pos.Score)
        x_test_neg = np.array(test_df_neg.drop("Score", axis=1))
        y_test_neg = np.array(test_df_neg.Score)

        x_test = np.concatenate((x_test_pos, x_test_neg))
        y_test = np.concatenate((y_test_pos, y_test_neg))

        _, _, f1_all_f8_lgb = GetProdictPS_lgb(X_train, Y_train, x_test, y_test)
        accuracy_pos_f8_lgb, _, _ = GetProdictPS_lgb(X_train, Y_train, x_test_pos, y_test_pos)

        accuracy_f8_lgb.append(accuracy_pos_f8_lgb)
        f1_f8_lgb.append(f1_all_f8_lgb)

        ###for pu f8
        _, _, f1_all_f8_pu = GetProdictPS_pu(x_test, X_train, y_test, Y_train)
        accuracy_pos_f8_pu, _, _ = GetProdictPS_pu(x_test_pos, X_train, y_test_pos, Y_train)

        accuracy_f8_pu.append(accuracy_pos_f8_pu)
        f1_f8_pu.append(f1_all_f8_pu)

        ###for lgb f1
        test_df_pos_f1 = test_df_pos.iloc[:, 7:9]
        train_df_pos_f1 = train_df_pos.iloc[:, 7:9]
        test_df_neg_f1 = test_df_neg.iloc[:, 7:9]
        train_df_neg_f1 = train_df_neg.iloc[:, 7:9]

        training_df_f1 = pd.concat([train_df_pos_f1, train_df_neg_f1])
        X_train_f1 = np.array(training_df_f1.drop("Score", axis=1))
        Y_train_f1 = np.array(training_df_f1.Score)

        x_test_pos_f1 = np.array(test_df_pos_f1.drop("Score", axis=1))
        y_test_pos_f1 = np.array(test_df_pos_f1.Score)
        x_test_neg_f1 = np.array(test_df_neg_f1.drop("Score", axis=1))
        y_test_neg_f1 = np.array(test_df_neg_f1.Score)

        x_test_f1 = np.concatenate((x_test_pos_f1, x_test_neg_f1))
        y_test_f1 = np.concatenate((y_test_pos_f1, y_test_neg_f1))

        _, _, f1_all_f1_lgb = GetProdictPS_lgb(X_train_f1, Y_train_f1, x_test_f1, y_test_f1)
        accuracy_pos_f1_lgb, _, _ = GetProdictPS_lgb(X_train_f1, Y_train_f1, x_test_pos_f1, y_test_pos_f1)

        accuracy_f1_lgb.append(accuracy_pos_f1_lgb)
        f1_f1_lgb.append(f1_all_f1_lgb)

        ###for lgb f7
        test_df_pos_f7 = test_df_pos.drop("PhosFreq", axis=1)
        train_df_pos_f7 = train_df_pos.drop("PhosFreq", axis=1)
        test_df_neg_f7 = test_df_neg.drop("PhosFreq", axis=1)
        train_df_neg_f7 = train_df_neg.drop("PhosFreq", axis=1)

        training_df_f7 = pd.concat([train_df_pos_f7, train_df_neg_f7])
        X_train_f7 = np.array(training_df_f7.drop("Score", axis=1))
        Y_train_f7 = np.array(training_df_f7.Score)

        x_test_pos_f7 = np.array(test_df_pos_f7.drop("Score", axis=1))
        y_test_pos_f7 = np.array(test_df_pos_f7.Score)
        x_test_neg_f7 = np.array(test_df_neg_f7.drop("Score", axis=1))
        y_test_neg_f7 = np.array(test_df_neg_f7.Score)

        x_test_f7 = np.concatenate((x_test_pos_f7, x_test_neg_f7))
        y_test_f7 = np.concatenate((y_test_pos_f7, y_test_neg_f7))

        _, _, f1_all_f7_lgb = GetProdictPS_lgb(X_train_f7, Y_train_f7, x_test_f7, y_test_f7)
        accuracy_pos_f7_lgb, _, _ = GetProdictPS_lgb(X_train_f7, Y_train_f7, x_test_pos_f7, y_test_pos_f7)

        accuracy_f7_lgb.append(accuracy_pos_f7_lgb)
        f1_f7_lgb.append(f1_all_f7_lgb)

    accuracy_f8_lgb_re10[re10] = accuracy_f8_lgb
    accuracy_f8_pu_re10[re10] = accuracy_f8_pu
    accuracy_f1_lgb_re10[re10] = accuracy_f1_lgb
    accuracy_f7_lgb_re10[re10] = accuracy_f7_lgb

    f1_f8_lgb_re10[re10], f1_f8_pu_re10[re10], f1_f1_lgb_re10[re10], f1_f7_lgb_re10[re10] = (
        f1_f8_lgb, f1_f8_pu, f1_f1_lgb, f1_f7_lgb)

accuracy_mean_f8_lgb = np.array(accuracy_f8_lgb_re10.values.mean())
accuracy_mean_f8_pu = np.array(accuracy_f8_pu_re10.values.mean())
accuracy_mean_f1_lgb = np.array(accuracy_f1_lgb_re10.values.mean())
accuracy_mean_f7_lgb = np.array(accuracy_f7_lgb_re10.values.mean())

f1_score_mean_f8_lgb = np.array(f1_f8_lgb_re10.values.mean())
f1_score_mean_f8_pu = np.array(f1_f8_pu_re10.values.mean())
f1_score_mean_f1_lgb = np.array(f1_f1_lgb_re10.values.mean())
f1_score_mean_f7_lgb = np.array(f1_f7_lgb_re10.values.mean())

accuracy_mean = np.transpose(np.array([accuracy_mean_f1_lgb, accuracy_mean_f7_lgb, accuracy_mean_f8_lgb, accuracy_mean_f8_pu]))
f1_score_mean = np.transpose(np.array([f1_score_mean_f1_lgb, f1_score_mean_f7_lgb, f1_score_mean_f8_lgb, f1_score_mean_f8_pu]))


Auu = pd.DataFrame({
                "TPR": accuracy_mean.flatten(),
                "F1score": f1_score_mean.flatten()
            }, columns=["TPR", "F1score"], index=["F1_lgb", "F7_lgb", "F8_lgb", "F8_pu"])
            
Auu.to_csv("PPF_known_Figure3c.csv")


