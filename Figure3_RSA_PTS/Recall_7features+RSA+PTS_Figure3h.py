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

def random_undersampling(tmp_df, TARGET_LABEL):
    df_majority = tmp_df[tmp_df[TARGET_LABEL] == 0]
    df_minority = tmp_df[tmp_df[TARGET_LABEL] == 1]
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority),  # to match minority class
                                       random_state=None)  # reproducible results
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    print("Undersampling complete!")
    print(df_downsampled[TARGET_LABEL].value_counts())
    return df_downsampled

Nops_df_org = pd.read_csv("NonPS_7features+RSA+PTS.csv", index_col=0)
Ps_df_org = pd.read_csv("PS_7features+RSA+PTS.csv", index_col=0)

mean_values_ps = Ps_df_org.mean()
Pas_all_filled = Ps_df_org.fillna(mean_values_ps)
mean_values_no = Nops_df_org.mean()
Nops_all_filled = Nops_df_org.fillna(mean_values_no)

#sample_num, hidden_size = 400, 200
sample_num, hidden_size = 800, 400
accuracy_df, recall_df, f1_score_df = (
    pd.DataFrame(data=None),
    pd.DataFrame(data=None),
    pd.DataFrame(data=None))
accuracy_df_new, recall_df_new, f1_score_df_new = (
    pd.DataFrame(data=None),
    pd.DataFrame(data=None),
    pd.DataFrame(data=None))

for re10 in range(0, 10):
    Nops_df_re = Nops_all_filled.sample(n=len(Pas_all_filled), random_state=re10)

    test_df_pos_raw = Pas_all_filled.sample(n=sample_num, random_state=re10)
    test_df_neg_raw = Nops_df_re.sample(n=sample_num, random_state=re10)
    df_raw = pd.concat([test_df_pos_raw, test_df_neg_raw])
    print(df_raw.Score.value_counts())

    df_downsampled = random_undersampling(df_raw, 'Score')
    df_downsampled = df_downsampled.sample(frac=1)
    df_downsampled = df_downsampled.reset_index()
    df_downsampled = df_downsampled.drop(columns=['UniprotEntry'])
    df = df_downsampled.copy()
    NON_LBL = [c for c in df.columns if c != 'Score']
    X = df[NON_LBL]
    y = df['Score']

    # Save the original labels and indices
    y_orig = y.copy()
    original_idx = np.where(df_downsampled.Score == 1)

    # Here we are imputing 400 positives as negative
    y.loc[np.random.choice(y[y == 1].index, replace=False, size=hidden_size)] = 0
    y.value_counts()

    accuracy, precision, recall, f1_score_ = [], [], [], []
    accuracy_new, precision_new, recall_new, f1_score_new = [], [], [], []

    ### lgb
    classifier = LGBMClassifier(objective='binary', boosting_type='dart')
    rf = classifier
    rf.fit(X, y)

    accuracy_lgb = accuracy_score(y_orig, rf.predict(X))
    recall_lgb = recall_score(y_orig, rf.predict(X))
    f1_score_lgb = f1_score(y_orig, rf.predict(X))

    accuracy.append(accuracy_lgb)
    recall.append(recall_lgb)
    f1_score_.append(f1_score_lgb)

    accuracy_df[re10] = accuracy
    recall_df[re10] = recall
    f1_score_df[re10] = f1_score_

    ### pu
    cf = BaggingClassifierPU(
        classifier,
        n_estimators=100,
        max_samples=100,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1
    )
    cf.fit(X, y)

    accuracy_pu = accuracy_score(y_orig, cf.predict(X))
    recall_pu = recall_score(y_orig, cf.predict(X))
    f1_score_pu = f1_score(y_orig, cf.predict(X))

    accuracy_new.append(accuracy_pu)
    recall_new.append(recall_pu)
    f1_score_new.append(f1_score_pu)

    accuracy_df_new[re10] = accuracy_new
    recall_df_new[re10] = recall_new
    f1_score_df_new[re10] = f1_score_new


########################################

accuracy_mean_mod = np.array(accuracy_df.mean(axis=1))
accuracy_mean_pu = np.array(accuracy_df_new.mean(axis=1))
recall_mean_mod = np.array(recall_df.mean(axis=1))
recall_mean_pu = np.array(recall_df_new.mean(axis=1))
f1_score_mean_mod = np.array(f1_score_df.mean(axis=1))
f1_score_mean_pu = np.array(f1_score_df_new.mean(axis=1))

lgb_mean = np.transpose(np.array([recall_mean_mod, f1_score_mean_mod, accuracy_mean_mod]))
pu_mean = np.transpose(np.array([recall_mean_pu, f1_score_mean_pu, accuracy_mean_pu]))

Auu = pd.DataFrame({
                "lgb": lgb_mean.flatten(),
                "pu": pu_mean.flatten()
            }, columns=["lgb", "pu"], index=["Recall", "F1_score", "Accuracy"])
Auu.to_csv("Recall_9features.csv")

