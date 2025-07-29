"""
Integrated Protein Classification PU Learning Pipeline
Complete workflow including data preprocessing, model training, scoring and reliability classification
"""

import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score


def step1_data_preprocessing_and_lgb_scoring():
    """
    STEP 1-2: Data preprocessing and LightGBM scoring
    """
    print("=== STEP 1-2: Data preprocessing and LightGBM scoring ===")
    
    # Read data lists
    nops_list = pd.read_csv("nops_list.csv", header=None)
    ps_list = pd.read_csv("speci8_list.csv", header=None)
    
    # Initialize data containers
    nops = []
    pas = []
    nops_all = pd.DataFrame()
    pas_all = pd.DataFrame()
    
    # Read all data files
    for i in range(len(ps_list)):
        pas.append(pd.read_csv(str((ps_list.iloc[i]).values[0]), index_col=0))
        nops.append(pd.read_csv(str((nops_list.iloc[i]).values[0]), index_col=0))
    
    # Concatenate all data
    for i in nops:
        nops_all = pd.concat([nops_all, i])
    
    for i in pas:
        pas_all = pd.concat([pas_all, i])
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'boosting_type': 'dart'
    }
    
    # Save concatenated data
    nops_all.to_csv("Nops_all.csv")
    pas_all.to_csv("Pas_all.csv")
    
    # Data imputation (fill missing values with mean)
    mean_values_ps = pas_all.mean()
    pas_all_filled = pas_all.fillna(mean_values_ps)
    
    mean_values_no = nops_all.mean()
    nops_all_filled = nops_all.fillna(mean_values_no)
    
    # Dataset splitting
    positive_samples_ = pas_all_filled.sample(n=6000, random_state=0)
    positive_index = positive_samples_.index
    positive_test = pas_all_filled.drop(labels=positive_index)
    positive_test.to_csv("positive_test.csv")
    positive_samples_.to_csv("positive_samples_.csv")
    
    background_samples_ = nops_all_filled.sample(n=144000, random_state=1)
    background_index = background_samples_.index
    background_test = nops_all_filled.drop(labels=background_index)
    background_test.to_csv("background_test.csv")
    background_samples_.to_csv("background_samples_.csv")
    
    # Initialize protein predictions dictionary
    protein_predictions = {}
    for i in background_index:
        protein_predictions[i] = []
    
    positive_samples = np.array(positive_samples_.drop("Score", axis=1))
    background_samples = np.array(background_samples_.drop("Score", axis=1))
    
    # Cross-validation training and prediction
    for i in range(0, len(background_samples), len(positive_samples)):
        neg_train_samples = background_samples[i:i+len(positive_samples)]
        train_samples = np.concatenate((positive_samples, neg_train_samples), axis=0)
        
        lgb_train = lgb.Dataset(train_samples, label=np.concatenate((np.ones(len(positive_samples)),
                                                                     np.zeros(len(neg_train_samples)))))
        model = lgb.train(params, lgb_train)
        
        rest_neg_index = np.delete(background_index, np.arange(i, i + len(positive_samples)))
        rest_neg_samples = np.delete(background_samples, np.arange(i, i + len(positive_samples)), axis=0)
        
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
    
    # Save LightGBM prediction scores
    output_file = 'LGB_prediction_score.csv'
    with open(output_file, 'w') as f:
        f.write("ProteinIndex,AveragePrediction\n")
        for protein_index, average_prediction in average_protein_predictions.items():
            f.write(f"{protein_index},{average_prediction}\n")
    
    print("Step 1-2 completed: LightGBM scoring finished")
    return True


def step3_pu_learning_with_oob_scoring():
    """
    STEP 3: PU learning with Out-of-Bag scoring
    """
    print("=== STEP 3: PU learning with OOB scoring ===")
    
    # Load datasets
    positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)
    background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
    background_index = background_samples_.index
    
    # Load validation data
    positive_valid = pd.read_csv("positive_test.csv", index_col=0)
    background_valid = pd.read_csv("background_test.csv", index_col=0)
    background_valid_ = background_valid.sample(n=len(positive_valid), random_state=2024)
    data_valid = pd.concat([positive_valid, background_valid_])
    
    # Prepare training data
    train_data = pd.concat([positive_samples_, background_samples_])
    X = train_data.drop('Score', axis=1).values
    y = train_data['Score'].values
    train_index = train_data.index
    
    # Train PU learning model with OOB scoring
    base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', random_state=2)
    cf = BaggingClassifierPU(base_estimator, n_estimators=100, max_samples=100, 
                           bootstrap=True, oob_score=True, n_jobs=-1)
    cf.fit(X, y)
    pu_score = cf.oob_decision_function_[:, 1]
    
    # Save OOB scores
    pred_results = pd.DataFrame({
        "ProteinIndex": train_index,
        "OOB_score": pu_score
    }, columns=["ProteinIndex", "OOB_score"])
    
    background_protein_puscore = pred_results.tail(144000)
    background_protein_puscore.to_csv("OOB_score.csv", index=False)
    
    print("Step 3 completed: OOB scoring finished")
    return True


def step4_reliability_classification_lgb():
    """
    STEP 4_1: Reliability classification using LightGBM scores
    """
    print("=== STEP 4_1: Reliability classification using LightGBM scores ===")
    
    # Load data
    positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)
    positive_valid = pd.read_csv("positive_test.csv", index_col=0)
    background_valid = pd.read_csv("background_test.csv", index_col=0)
    background_valid_ = background_valid.sample(n=len(positive_valid), random_state=2024)
    data_valid = pd.concat([positive_valid, background_valid_])
    
    average_protein_predictions = pd.read_csv("LGB_prediction_score.csv", index_col=0)
    background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
    background_index = background_samples_.index
    
    # Define condition thresholds
    condition_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ACC_data = pd.DataFrame(index=condition_, columns=condition_)
    F1S_data = pd.DataFrame(index=condition_, columns=condition_)
    
    # Grid search over threshold combinations
    for mmm in range(9):
        # Filter negative samples based on lower threshold
        filtered_data = average_protein_predictions[
            average_protein_predictions['AveragePrediction'] < condition_[mmm]]
        X_train_neg_final_index = filtered_data.index
        X_train_neg_final = background_samples_[background_index.isin(X_train_neg_final_index)]
        
        for nnn in range(mmm, 9):
            # Filter positive samples based on upper threshold
            filtered_data_ps = average_protein_predictions[
                average_protein_predictions['AveragePrediction'] > condition_[nnn]]
            ps_final_index = filtered_data_ps.index
            ps_final_data = background_samples_[background_index.isin(ps_final_index)]
            ps_final_data['Score'] = 1
            
            negative_samples = X_train_neg_final
            positive_samples = pd.concat([positive_samples_, ps_final_data])
            
            # Train multiple models
            train_num = 7000
            for i in range(10):
                sampled_neg_data = negative_samples.sample(n=train_num)
                sampled_pos_data = positive_samples.sample(n=train_num)
                train_data = pd.concat([sampled_pos_data, sampled_neg_data])
                
                base_estimator = LGBMClassifier(
                    objective='binary', boosting_type='dart', learning_rate=0.5,
                    bagging_fraction=0.5, feature_fraction=0.8, 
                    min_child_samples=18, num_leaves=5, random_state=2
                )
                
                model = BaggingClassifierPU(base_estimator, random_state=2)
                model.fit(train_data.drop('Score', axis=1), train_data['Score'])
                joblib.dump(model, f'pu_bagging_model_{i + 1}.joblib')
            
            # Evaluate models
            accuracies, f1scores = [], []
            for i in range(10):
                model_file = f'pu_bagging_model_{i + 1}.joblib'
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
    
    print("LightGBM-based Accuracy Results:")
    print(ACC_data)
    print("\nLightGBM-based F1 Score Results:")
    print(F1S_data)
    
    # Save results
    ACC_data.to_csv("LGB_accuracy_results.csv")
    F1S_data.to_csv("LGB_f1_results.csv")
    
    return ACC_data, F1S_data


def step5_reliability_classification_pu():
    """
    STEP 4_2: Reliability classification using PU learning OOB scores
    """
    print("=== STEP 4_2: Reliability classification using PU learning OOB scores ===")
    
    # Load data
    positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0)
    positive_valid = pd.read_csv("positive_test.csv", index_col=0)
    background_valid = pd.read_csv("background_test.csv", index_col=0)
    background_valid_ = background_valid.sample(n=len(positive_valid), random_state=2024)
    data_valid = pd.concat([positive_valid, background_valid_])
    
    average_protein_predictions = pd.read_csv("OOB_score.csv", index_col=0)
    background_samples_ = pd.read_csv("background_samples_.csv", index_col=0)
    background_index = background_samples_.index
    
    # Define condition thresholds
    condition_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ACC_data = pd.DataFrame(index=condition_, columns=condition_)
    F1S_data = pd.DataFrame(index=condition_, columns=condition_)
    
    # Grid search over threshold combinations
    for mmm in range(9):
        # Filter negative samples based on lower threshold
        filtered_data = average_protein_predictions[
            average_protein_predictions['OOB_score'] < condition_[mmm]]
        X_train_neg_final_index = filtered_data.index
        X_train_neg_final = background_samples_[background_index.isin(X_train_neg_final_index)]
        
        for nnn in range(mmm, 9):
            # Filter positive samples based on upper threshold
            filtered_data_ps = average_protein_predictions[
                average_protein_predictions['OOB_score'] > condition_[nnn]]
            ps_final_index = filtered_data_ps.index
            ps_final_data = background_samples_[background_index.isin(ps_final_index)]
            ps_final_data['Score'] = 1
            
            negative_samples = X_train_neg_final
            positive_samples = pd.concat([positive_samples_, ps_final_data])
            
            # Train multiple models
            train_num = 6000
            for i in range(10):
                sampled_neg_data = negative_samples.sample(n=train_num)
                sampled_pos_data = positive_samples.sample(n=train_num)
                train_data = pd.concat([sampled_pos_data, sampled_neg_data])
                
                base_estimator = LGBMClassifier(
                    objective='binary', boosting_type='dart', learning_rate=0.5,
                    bagging_fraction=0.5, feature_fraction=0.8,
                    min_child_samples=18, num_leaves=5, random_state=2
                )
                
                model = BaggingClassifierPU(base_estimator, random_state=2)
                model.fit(train_data.drop('Score', axis=1), train_data['Score'])
                joblib.dump(model, f'pu_bagging_model222_{i + 1}.joblib')
            
            # Evaluate models
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
    
    print("PU learning-based Accuracy Results:")
    print(ACC_data)
    print("\nPU learning-based F1 Score Results:")
    print(F1S_data)
    
    # Save results
    ACC_data.to_csv("PU_accuracy_results.csv")
    F1S_data.to_csv("PU_f1_results.csv")
    
    return ACC_data, F1S_data


def main():
    """
    Main function to run the complete pipeline
    """
    print("Starting Protein Classification PU Learning Pipeline")
    print("=" * 60)
    
    try:
        # Step 1-2: Data preprocessing and LightGBM scoring
        step1_data_preprocessing_and_lgb_scoring()
        
        # Step 3: PU learning with OOB scoring
        step3_pu_learning_with_oob_scoring()
        
        # Step 4-1: Reliability classification using LightGBM scores
        lgb_acc, lgb_f1 = step4_reliability_classification_lgb()
        
        # Step 4-2: Reliability classification using PU learning OOB scores
        pu_acc, pu_f1 = step5_reliability_classification_pu()
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("Results saved to respective CSV files.")
        
    except Exception as e:
        print(f"Error occurred during pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
