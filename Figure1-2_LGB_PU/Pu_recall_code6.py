import numpy as np
import pandas as pd
import math
import sys
import time
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle, resample
from baggingPU import BaggingClassifierPU
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix


def getClassifier(model):
    if model == "SVM":
        from sklearn.svm import SVC
        return SVC(kernel='rbf', gamma='auto', random_state=0)
    elif model == "DT":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier()
    elif model == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=0)
    elif model == "LGB":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective='binary', boosting_type='dart')
    elif model == "NB":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif model == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam')

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


##########################################

Nops_list = pd.read_csv("nops_list.csv", header=None)
Ps_list = pd.read_csv("speci8_list.csv", header=None)

Pas = []
Pas_all = pd.DataFrame()
for i in range(0, len(Ps_list)):
    Pas.append(pd.read_csv(str((Ps_list.iloc[i]).values[0]), index_col=0))
for i in Pas:
    Pas_all = pd.concat([Pas_all, i])
    
#
Pas_all_filled = Pas_all.fillna(0)

models = ["RF", "LGB", "DT", "SVM", "NB", "MLP"]
sample_num, hidden_size = 800, 400

accuracy_df, recall_df, f1_score_df = (
    pd.DataFrame(data=None, index=models),
    pd.DataFrame(data=None, index=models),
    pd.DataFrame(data=None, index=models))
accuracy_df_new, recall_df_new, f1_score_df_new = (
    pd.DataFrame(data=None, index=models),
    pd.DataFrame(data=None, index=models),
    pd.DataFrame(data=None, index=models))
    
for re in range(0, 10):
    Nops = []
    Nops_all = pd.DataFrame()

    for nn in range(0, len(Ps_list)):
        Nops.append((pd.read_csv(str((Nops_list.iloc[nn]).values[0]), index_col=0)).sample(n=len(Pas[nn]), random_state=re))

    for nnn in Nops:
        Nops_all = pd.concat([Nops_all, nnn])

    Nops_all_filled = Nops_all.fillna(0)

    test_df_pos_raw = Pas_all_filled.sample(n=sample_num, random_state=re)
    test_df_neg_raw = Nops_all_filled.sample(n=sample_num, random_state=re)
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

    for Mod in models:
        classifier = getClassifier(Mod)
        rf = classifier
        rf.fit(X, y)

        print('---- {} ----'.format(Mod))
        #print(print_cm(sklearn.metrics.confusion_matrix(y_orig, rf.predict(X)), labels=['Pre_negative', 'Pre_positive']))
        #cm = confusion_matrix(y_orig, rf.predict(X))
        #cm_df = pd.DataFrame(cm, index=['Actual_negative', 'Actual_positive'], columns=['Pre_negative', 'Pre_positive'])

        accuracy_mod = accuracy_score(y_orig, rf.predict(X))
        recall_mod = recall_score(y_orig, rf.predict(X))
        f1_score_mod = f1_score(y_orig, rf.predict(X))

        accuracy.append(accuracy_mod)
        recall.append(recall_mod)
        f1_score_.append(f1_score_mod)

    accuracy_df[re] = accuracy
    recall_df[re] = recall
    f1_score_df[re] = f1_score_

    for Mod in models:
        classifier = getClassifier(Mod)
        cf = BaggingClassifierPU(
            classifier,
            n_estimators=100,
            max_samples=sum(y),
            bootstrap=True,
            oob_score=True,
            n_jobs=-1
        )
        cf.fit(X, y)

        print('---- {} ----'.format(Mod))
        #print(print_cm(sklearn.metrics.confusion_matrix(y_orig, rf.predict(X)), labels=['Pre_negative', 'Pre_positive']))
        #cm = confusion_matrix(y_orig, cf.predict(X))
        #cm_df = pd.DataFrame(cm, index=['Actual_negative', 'Actual_positive'], columns=['Pre_negative', 'Pre_positive'])

        accuracy_pu = accuracy_score(y_orig, cf.predict(X))
        recall_pu = recall_score(y_orig, cf.predict(X))
        f1_score_pu = f1_score(y_orig, cf.predict(X))

        accuracy_new.append(accuracy_pu)
        recall_new.append(recall_pu)
        f1_score_new.append(f1_score_pu)

    accuracy_df_new[re] = accuracy_new
    recall_df_new[re] = recall_new
    f1_score_df_new[re] = f1_score_new

########################################

accuracy_mean_mod = np.array(accuracy_df.mean(axis=1))
accuracy_mean_pu = np.array(accuracy_df_new.mean(axis=1))
Auu = pd.DataFrame({
                "mod": accuracy_mean_mod,
                "pu": accuracy_mean_pu
            }, columns=["mod", "pu"], index=models)
Auu.to_csv("Accuracy_hide2025_newparameter.csv")

recall_mean_mod = np.array(recall_df.mean(axis=1))
recall_mean_pu = np.array(recall_df_new.mean(axis=1))
Rec = pd.DataFrame({
                "mod": recall_mean_mod,
                "pu": recall_mean_pu
            }, columns=["mod", "pu"], index=models)
Rec.to_csv("Recall_hide2025_newparameter.csv")

f1_score_mean_mod = np.array(f1_score_df.mean(axis=1))
f1_score_mean_pu = np.array(f1_score_df_new.mean(axis=1))
F1s = pd.DataFrame({
                "mod": f1_score_mean_mod,
                "pu": f1_score_mean_pu
            }, columns=["mod", "pu"], index=models)
F1s.to_csv("F1score_hide2025_newparameter.csv")



