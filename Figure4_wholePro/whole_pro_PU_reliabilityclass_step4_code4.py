### STEP4 ###

import numpy as np
from lightgbm import LGBMClassifier
import pandas as pd
import os
from baggingPU import BaggingClassifierPU
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score

#positive training data
positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)
verified_samples = pd.read_csv("verified_samples.csv", index_col=0)
positive_samples_700 = positive_samples_.sample(n=len(verified_samples), random_state=2025)
#
positive_samples_combined = pd.concat([positive_samples_700, verified_samples])

#positive validation data
positive_valid = pd.read_csv("positive_test.csv", index_col=0)
verified_remaining = pd.read_csv("verified_test.csv", index_col=0)
positive_valid_55 = positive_valid.sample(n=len(verified_remaining), random_state=2025)
#
positive_valid_all = pd.concat([positive_valid_55, verified_remaining])

#negative data
negative_valid_ = pd.read_csv("CorrectedData_negative_2121.csv", index_col=0)
mean_negative_valid = negative_valid_.mean()
negative_valid = negative_valid_.fillna(mean_negative_valid)

negative_valid_all = negative_valid.sample(n=len(positive_valid_all), random_state=2024) 
negative_train = negative_valid.sample(n=len(positive_samples_combined), random_state=2025)

data_valid = pd.concat([positive_valid_all, negative_valid_all])

average_protein_predictions = pd.read_csv("PU_prediction_score.csv", index_col=0)
background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
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
        positive_samples = ps_final_data

        #
        train_predict_num = 700
        for i in range(10):
            if train_predict_num <= len(positive_samples):
                sampled_pos_data = positive_samples.sample(n=train_predict_num, replace=False)

            else:
                sampled_pos_data = positive_samples.sample(n=train_predict_num, replace=True)
                
            if train_predict_num <= len(positive_samples):
                sampled_neg_data = negative_samples.sample(n=train_predict_num, replace=False)

            else:
                sampled_neg_data = negative_samples.sample(n=train_predict_num, replace=True)

            train_positive = pd.concat([positive_samples_combined, sampled_pos_data]) #1400+700
            train_negative = pd.concat([negative_train, sampled_neg_data]) #1400+700

            train_data = pd.concat([train_positive, train_negative])

            base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', learning_rate=0.5,
                                            bagging_fraction=0.5,
                                            feature_fraction=0.8, min_child_samples=18, num_leaves=5, random_state=2)
            
            model = BaggingClassifierPU(base_estimator, random_state=2)
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
