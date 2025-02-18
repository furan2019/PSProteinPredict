### STEP3_2 ### AFTER STEP4

import numpy as np
from lightgbm import LGBMClassifier
import pandas as pd
import os
from baggingPU import BaggingClassifierPU
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score

positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)

positive_valid = pd.read_csv("positive_test.csv", index_col=0)
background_valid = pd.read_csv("background_test.csv", index_col=0)
background_valid_ = background_valid.sample(n=len(positive_valid), random_state=2024)
data_valid = pd.concat([positive_valid, background_valid_])

average_protein_predictions = pd.read_csv("OOB_score.csv", index_col=0)
background_samples_ = pd.read_csv("background_samples_.csv", index_col=0) #特征文件
background_index = background_samples_.index

condition_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ACC_data = pd.DataFrame(index=condition_, columns=condition_)
F1S_data = pd.DataFrame(index=condition_, columns=condition_)

##
for mmm in range(9):
    filtered_data = average_protein_predictions[average_protein_predictions['OOB_score'] < condition_[mmm]]
    X_train_neg_final_index = filtered_data.index
    X_train_neg_final = background_samples_[background_index.isin(X_train_neg_final_index)]

    for nnn in range(mmm, 9):
        filtered_data_ps = average_protein_predictions[
            average_protein_predictions['OOB_score'] > condition_[nnn]]
        ps_final_index = filtered_data_ps.index
        ps_final_data = background_samples_[background_index.isin(ps_final_index)]
        ps_final_data['Score'] = 1

        negative_samples = X_train_neg_final
        positive_samples = pd.concat([positive_samples_, ps_final_data])

        train_num = 6000
        for i in range(10):
            # sampled_neg_data = negative_samples.sample(n=train_num, random_state=i)
            # sampled_pos_data = positive_samples.sample(n=train_num, random_state=i)
            sampled_neg_data = negative_samples.sample(n=train_num)
            sampled_pos_data = positive_samples.sample(n=train_num)

            train_data = pd.concat([sampled_pos_data, sampled_neg_data])

            #base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', random_state = 2)
            base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', learning_rate=0.5,
                                            bagging_fraction=0.5,
                                            feature_fraction=0.8, min_child_samples=18, num_leaves=5, random_state=2)
            model = BaggingClassifierPU(base_estimator, random_state=2)
            
            #model = BaggingClassifierPU(base_estimator, n_estimators=100, max_samples=100, bootstrap=True, oob_score=True, n_jobs=-1)
            model.fit(train_data.drop('Score', axis=1), train_data['Score'])

            joblib.dump(model, f'pu_bagging_model222_{i + 1}.joblib')

        accuracies, f1scores = [], []
        for i in range(10):
            model_file = f'pu_bagging_model222_{i + 1}.joblib'
            if not os.path.exists(model_file):
                print(f"Model file {model_file} not found.")
                continue
            pu_bagging = joblib.load(model_file)

            pu_pred = pu_bagging.predict(positive_valid.drop('Score', axis=1))
            pu_pred_f1 = pu_bagging.predict(data_valid.drop('Score', axis=1))

            accuracy = accuracy_score(positive_valid['Score'], pu_pred)
            accuracies.append(accuracy)

            f1score = f1_score(data_valid['Score'], pu_pred_f1)
            f1scores.append(f1score)

        ACC_data.iloc[mmm, nnn] = np.mean(accuracies)
        F1S_data.iloc[mmm, nnn] = np.mean(f1scores)

print(ACC_data)
print(F1S_data) 
