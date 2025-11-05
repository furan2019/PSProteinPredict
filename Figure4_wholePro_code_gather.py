"""
Integrated Protein Classification PU Learning Pipeline
Complete workflow including data preprocessing, model training, scoring and reliability classification
"""

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib
import os
from baggingPU import BaggingClassifierPU

def step1_2_lgb_score():
    """STEP1-2: LightGBM scoring with corrected data"""
    print("Running STEP1-2: LightGBM scoring...")
    
    # Load data
    Nops_all = pd.read_csv("Nops_all.csv", index_col=0)
    Pas_all = pd.read_csv("Pas_all.csv", index_col=0)

    # Fill missing values with mean
    mean_values_ps = Pas_all.mean()
    Pas_all_filled = Pas_all.fillna(mean_values_ps)

    mean_values_no = Nops_all.mean()
    Nops_all_filled = Nops_all.fillna(mean_values_no)

    # Load verified positive samples and fill missing values
    verified_positive = pd.read_csv("CorrectedData_positive_755.csv", index_col=0)
    mean_values_verified_positive = verified_positive.mean()
    verified_positive_filled = verified_positive.fillna(mean_values_verified_positive)
    
    # Randomly select 700 verified positive samples for correction, remaining 55 as backup
    verified_samples = verified_positive_filled.sample(n=700, random_state=0)
    verified_remaining = verified_positive_filled.drop(verified_samples.index)
    verified_remaining.to_csv("verified_test.csv")
    verified_samples.to_csv("verified_samples.csv")

    # Sample positive data
    positive_samples_ = Pas_all_filled.sample(n=6000, random_state=0)
    positive_index = positive_samples_.index
    positive_test = Pas_all_filled.drop(labels=positive_index)
    positive_test.to_csv("positive_test.csv")
    positive_samples_.to_csv("positive_samples_.csv")

    # Oversample verified positive samples to match unverified positive samples 1:1
    verified_oversampled = resample(verified_samples, 
                                   replace=True, 
                                   n_samples=len(positive_samples_), 
                                   random_state=0)

    # Sample background data
    background_samples_ = Nops_all_filled.sample(n=144000, random_state=1)
    background_index = background_samples_.index
    background_test = Nops_all_filled.drop(labels=background_index)
    background_test.to_csv("background_test.csv")
    background_samples_.to_csv("background_samples_.csv")

    # Initialize protein predictions dictionary
    protein_predictions = {}
    for i in background_index:
        protein_predictions[i] = []

    # Combine oversampled verified positive samples and original unverified positive samples
    positive_samples_combined = pd.concat([positive_samples_, verified_oversampled])
    positive_samples = np.array(positive_samples_combined.drop("Score", axis=1))
    background_samples = np.array(background_samples_.drop("Score", axis=1))

    # Model parameters
    params = {
        'objective': 'binary',
        'boosting_type': 'dart'
    }

    # Training and prediction loop
    for i in range(0, len(background_samples), len(positive_samples_combined)):
        neg_train_samples = background_samples[i:i+len(positive_samples_combined)]
        train_samples = np.concatenate((positive_samples, neg_train_samples), axis=0)

        lgb_train = lgb.Dataset(train_samples, label=np.concatenate((np.ones(len(positive_samples)),
                                                                     np.zeros(len(neg_train_samples)))))
        model = lgb.train(params, lgb_train)
        
        rest_neg_index = np.delete(background_index, np.arange(i, i + len(positive_samples_combined)))
        rest_neg_samples = np.delete(background_samples, np.arange(i, i + len(positive_samples_combined)), axis=0)

        rest_predictions = model.predict(rest_neg_samples)
        
        for j, prot_idx in enumerate(rest_neg_index):
            protein_predictions[prot_idx].append(rest_predictions[j])

    # Calculate average predictions
    average_protein_predictions = {}
    for protein_index, predictions in protein_predictions.items():
        if len(predictions) > 0:
            average_prediction = np.mean(predictions)
        else:
            average_prediction = 0.0
        average_protein_predictions[protein_index] = average_prediction

    # Save results
    output_file = 'LGB_prediction_score_CorrectionData.csv'
    with open(output_file, 'w') as f:
        f.write("ProteinIndex,LGBscore\n")
        for protein_index, average_prediction in average_protein_predictions.items():
            f.write(f"{protein_index},{average_prediction}\n")
    
    print("STEP1-2 completed: LGB_prediction_score.csv saved")

def step3_pu_score():
    """STEP3: PU learning scoring with bagging classifier"""
    print("Running STEP3: PU learning scoring...")
    
    # Load verified positive samples
    verified_samples = pd.read_csv("verified_samples.csv", index_col=0)

    # Load positive and background datasets
    positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)
    background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
    background_index = background_samples_.index

    # Oversample verified positive samples to match unverified positive samples 1:1
    verified_oversampled = resample(verified_samples, 
                                   replace=True, 
                                   n_samples=len(positive_samples_), 
                                   random_state=0)
    
    # Combine oversampled verified positive samples and original unverified positive samples
    positive_samples_combined = pd.concat([positive_samples_, verified_oversampled])
                               
    # Prepare validation data
    positive_valid = pd.read_csv("positive_test.csv", index_col=0)
    background_valid = pd.read_csv("background_test.csv", index_col=0)
    background_valid_ = background_valid.sample(n=len(positive_valid), random_state=2024)
    data_valid = pd.concat([positive_valid, background_valid_])

    # Prepare training data
    train_data = pd.concat([positive_samples_combined, background_samples_])
    X = train_data.drop('Score', axis=1).values
    y = train_data['Score'].values
    train_index = train_data.index

    # Train PU bagging classifier
    base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', random_state=2)
    cf = BaggingClassifierPU(base_estimator, n_estimators=100, max_samples=100, bootstrap=True, oob_score=True, n_jobs=-1)
    cf.fit(X, y)
    pu_score = cf.oob_decision_function_[:, 1]

    # Save OOB scores
    predResults = pd.DataFrame({
        "ProteinIndex": train_index,
        "OOB_score": pu_score
    }, columns=["ProteinIndex", "OOBscore"])

    background_protein_puscore = predResults.tail(144000)
    background_protein_puscore.to_csv("PU_prediction_score.csv", index=False)
    
    print("STEP3 completed: PU_prediction_score.csv saved")

def step4_1_lgb_reliability_classification():
    """STEP4_1: Reliability classification using LGB prediction scores"""
    print("Running STEP4_1: Reliability classification with LGB scores...")
    
    # Prepare positive training set: plant positive samples + verified positive samples (total 1400)
    positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)  # 6000 samples
    verified_samples = pd.read_csv("verified_samples.csv", index_col=0)  # 700 samples
    
    # Sample unverified to match verified positive samples 1:1
    positive_samples_700 = positive_samples_.sample(n=len(verified_samples), random_state=2025)
    # Combine total 1400
    positive_samples_combined = pd.concat([positive_samples_700, verified_samples])

    # Prepare positive validation set: total 110 samples
    positive_valid = pd.read_csv("positive_test.csv", index_col=0)  # 559 samples
    verified_remaining = pd.read_csv("verified_test.csv", index_col=0)  # 55 samples
    
    # Sample unverified to match verified positive samples 1:1
    positive_valid_55 = positive_valid.sample(n=len(verified_remaining), random_state=2025)
    # Combine total 110 samples
    positive_valid_all = pd.concat([positive_valid_55, verified_remaining])

    # Prepare negative dataset: 1400 training samples, 110 validation samples
    negative_valid_ = pd.read_csv("CorrectedData_negative_2121.csv", index_col=0)  # 2121 samples
    mean_negative_valid = negative_valid_.mean()
    negative_valid = negative_valid_.fillna(mean_negative_valid)

    negative_valid_all = negative_valid.sample(n=len(positive_valid_all), random_state=2024)  # 110 samples
    negative_train = negative_valid.sample(n=len(positive_samples_combined), random_state=2025)  # 1400 samples

    # Combine positive and negative validation data
    data_valid = pd.concat([positive_valid_all, negative_valid_all])

    # Load LGB prediction scores
    average_protein_predictions = pd.read_csv("LGB_prediction_score.csv", index_col=0)

    # Load background dataset (feature file)
    background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
    background_index = background_samples_.index

    # Threshold conditions
    condition_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ACC_data = pd.DataFrame(index=condition_, columns=condition_)
    F1S_data = pd.DataFrame(index=condition_, columns=condition_)

    # Grid search over thresholds
    for mmm in range(9):
        filtered_data = average_protein_predictions[average_protein_predictions['AveragePrediction'] < condition_[mmm]]
        X_train_neg_final_index = filtered_data.index
        X_train_neg_final = background_samples_[background_index.isin(X_train_neg_final_index)]  # Predicted negative samples with scores

        for nnn in range(mmm, 9):
            filtered_data_ps = average_protein_predictions[
                average_protein_predictions['AveragePrediction'] > condition_[nnn]]
            ps_final_index = filtered_data_ps.index
            ps_final_data = background_samples_[background_index.isin(ps_final_index)]  # Predicted positive samples with scores
            ps_final_data['Score'] = 1

            negative_samples = X_train_neg_final
            positive_samples = ps_final_data

            # Training loop
            train_predict_num = 700
            for i in range(10):
                if train_predict_num <= len(positive_samples):
                    sampled_pos_data = positive_samples.sample(n=train_predict_num, replace=False)
                else:
                    sampled_pos_data = positive_samples.sample(n=train_predict_num, replace=True)
                    
                if train_predict_num <= len(negative_samples):
                    sampled_neg_data = negative_samples.sample(n=train_predict_num, replace=False)
                else:
                    sampled_neg_data = negative_samples.sample(n=train_predict_num, replace=True)
                    
                train_positive = pd.concat([positive_samples_combined, sampled_pos_data])  # 1400 + 700
                train_negative = pd.concat([negative_train, sampled_neg_data])  # 1400 + 700

                train_data = pd.concat([train_positive, train_negative])

                base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', learning_rate=0.5,
                                                bagging_fraction=0.5, feature_fraction=0.8, 
                                                min_child_samples=18, num_leaves=5, random_state=2)

                model = BaggingClassifierPU(base_estimator, random_state=2)
                model.fit(train_data.drop('Score', axis=1), train_data['Score'])

                joblib.dump(model, f'pu_bagging_model_{i + 1}.joblib')

            # Evaluation
            accuracies, f1scores = [], []
            for i in range(10):
                model_file = f'pu_bagging_model_{i + 1}.joblib'
                if not os.path.exists(model_file):
                    print(f"Model file {model_file} not found.")
                    continue
                pu_bagging = joblib.load(model_file)

                pu_pred = pu_bagging.predict(positive_valid_all.drop('Score', axis=1))
                pu_pred_f1 = pu_bagging.predict(data_valid.drop('Score', axis=1))

                accuracy = accuracy_score(positive_valid_all['Score'], pu_pred)
                accuracies.append(accuracy)

                f1score = f1_score(data_valid['Score'], pu_pred_f1)
                f1scores.append(f1score)

            ACC_data.iloc[mmm, nnn] = np.mean(accuracies)
            F1S_data.iloc[mmm, nnn] = np.mean(f1scores)

    print("Accuracy Results:")
    print(ACC_data)
    print("F1 Score Results:")
    print(F1S_data)
    
    print("STEP4_1 completed")

def step4_2_pu_reliability_classification():
    """STEP4_2: Reliability classification using PU OOB scores"""
    print("Running STEP4_2: Reliability classification with PU OOB scores...")
    
    # Prepare positive training set: plant positive samples + verified positive samples (total 1400)
    positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)  # 6000 samples
    verified_samples = pd.read_csv("verified_samples.csv", index_col=0)  # 700 samples
    
    # Sample unverified to match verified positive samples 1:1
    positive_samples_700 = positive_samples_.sample(n=len(verified_samples), random_state=2025)
    # Combine total 1400
    positive_samples_combined = pd.concat([positive_samples_700, verified_samples])

    # Prepare positive validation set: total 110 samples
    positive_valid = pd.read_csv("positive_test.csv", index_col=0)  # 559 samples
    verified_remaining = pd.read_csv("verified_test.csv", index_col=0)  # 55 samples
    
    # Sample unverified to match verified positive samples 1:1
    positive_valid_55 = positive_valid.sample(n=len(verified_remaining), random_state=2025)
    # Combine total 110 samples
    positive_valid_all = pd.concat([positive_valid_55, verified_remaining])

    # Prepare negative dataset: 1400 training samples, 110 validation samples
    negative_valid_ = pd.read_csv("CorrectedData_negative_2121.csv", index_col=0)  # 2121 samples
    mean_negative_valid = negative_valid_.mean()
    negative_valid = negative_valid_.fillna(mean_negative_valid)

    negative_valid_all = negative_valid.sample(n=len(positive_valid_all), random_state=2024)  # 110 samples
    negative_train = negative_valid.sample(n=len(positive_samples_combined), random_state=2025)  # 1400 samples

    # Combine positive and negative validation data
    data_valid = pd.concat([positive_valid_all, negative_valid_all])

    # Load PU OOB scores
    average_protein_predictions = pd.read_csv("PU_prediction_score.csv", index_col=0)

    # Load background dataset (feature file)
    background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
    background_index = background_samples_.index

    # Threshold conditions
    condition_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ACC_data = pd.DataFrame(index=condition_, columns=condition_)
    F1S_data = pd.DataFrame(index=condition_, columns=condition_)

    # Grid search over thresholds
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

            # Training loop
            train_predict_num = 700
            for i in range(10):
                if train_predict_num <= len(positive_samples):
                    sampled_pos_data = positive_samples.sample(n=train_predict_num, replace=False)
                else:
                    sampled_pos_data = positive_samples.sample(n=train_predict_num, replace=True)
                    
                if train_predict_num <= len(negative_samples):
                    sampled_neg_data = negative_samples.sample(n=train_predict_num, replace=False)
                else:
                    sampled_neg_data = negative_samples.sample(n=train_predict_num, replace=True)

                train_positive = pd.concat([positive_samples_combined, sampled_pos_data])  # 1400 + 700
                train_negative = pd.concat([negative_train, sampled_neg_data])  # 1400 + 700

                train_data = pd.concat([train_positive, train_negative])

                base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', learning_rate=0.5,
                                                bagging_fraction=0.5, feature_fraction=0.8, 
                                                min_child_samples=18, num_leaves=5, random_state=2)
                
                model = BaggingClassifierPU(base_estimator, random_state=2)
                model.fit(train_data.drop('Score', axis=1), train_data['Score'])

                joblib.dump(model, f'pu_bagging_model222_{i + 1}.joblib')

            # Evaluation
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

    print("Accuracy Results:")
    print(ACC_data)
    print("F1 Score Results:")
    print(F1S_data)
    
    print("STEP4_2 completed")

def main():
    """Main function to run all steps"""
    print("Starting protein classification pipeline...")
    
    # Run STEP1-2
    step1_2_lgb_score()
    
    # Run STEP3  
    step3_pu_score()
    
    # Run STEP4_1
    step4_1_lgb_reliability_classification()
    
    # Run STEP4_2
    step4_2_pu_reliability_classification()
    
    print("All steps completed!")

if __name__ == "__main__":
    main()
