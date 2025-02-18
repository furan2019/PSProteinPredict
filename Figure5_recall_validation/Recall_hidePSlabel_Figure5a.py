import numpy as np
import pandas as pd
import os
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
        
# ps data set
positive_samples_08 = pd.read_csv("unlabel_2_ps.csv", index_col=0)
positive_samples_or = positive_samples_08.sample(n=20000, random_state=4048)
mean_values_ps = positive_samples_or.mean()
positive_samples = positive_samples_or.fillna(mean_values_ps)

# nonps data set
negative_samples_01 = pd.read_csv("unlabel_2_nops.csv", index_col=0)
negative_samples_or = negative_samples_01.sample(n=20000, random_state=4048)
mean_values_no = negative_samples_or.mean()
negative_samples = negative_samples_or.fillna(mean_values_no)

#training data
train_data = pd.concat([positive_samples, negative_samples])
X = train_data.drop('Score', axis=1).values
y = train_data['Score'].values

#test data
X_train_or, X_pred_or, y_train_or, y_pred_or = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

max_mislabeled_positive = 14000

PN_model = LGBMClassifier(objective='binary', boosting_type='dart')
PU_model = BaggingClassifierPU(
    PN_model,
    n_estimators=100,
    max_samples=100,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1)
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam') 
PU_rf = BaggingClassifierPU(
    rf_model,
    n_estimators=100,
    max_samples=100,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1)
PU_mlp = BaggingClassifierPU(
    mlp_model,
    n_estimators=100,
    max_samples=100,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1)
    
    
#evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred),accuracy_score(y_test, y_pred),recall_score(y_test, y_pred)

for i in range(0, max_mislabeled_positive + 1000, 1000):
    mislabeled_positives = min(i, 20000)
    
    mislabeled_indices = np.random.choice(np.where(y_train_or == 1)[0], mislabeled_positives, replace=False)
    
    y_train_mislabeled = np.copy(y_train_or)
    y_train_mislabeled[mislabeled_indices] = 0
    
    lgbm_f1score = evaluate_model(PN_model, X_train_or, y_train_mislabeled, X_pred_or, y_pred_or)
    pu_f1score = evaluate_model(PU_model, X_train_or, y_train_mislabeled, X_pred_or, y_pred_or)
    
    print(f"Iteration {i//1000}, Mislabeled Postives: {mislabeled_positives}, PN Accuracy: {lgbm_f1score}")
    print(f"Iteration {i//1000}, Mislabeled Postives: {mislabeled_positives}, PU Accuracy: {pu_f1score}")
    
    rf_f1score = evaluate_model(rf_model, X_train_or, y_train_mislabeled, X_pred_or, y_pred_or)
    pu_rf_f1score = evaluate_model(PU_rf, X_train_or, y_train_mislabeled, X_pred_or, y_pred_or)
    
    print(f"Iteration {i//1000}, Mislabeled Postives: {mislabeled_positives}, rf Accuracy: {rf_f1score}")
    print(f"Iteration {i//1000}, Mislabeled Postives: {mislabeled_positives}, PU_rf Accuracy: {pu_rf_f1score}")
    
    mlp_f1score = evaluate_model(mlp_model, X_train_or, y_train_mislabeled, X_pred_or, y_pred_or)
    pu_mlp_f1score = evaluate_model(PU_mlp, X_train_or, y_train_mislabeled, X_pred_or, y_pred_or)
    
    print(f"Iteration {i//1000}, Mislabeled Postives: {mislabeled_positives}, mlp Accuracy: {mlp_f1score}")
    print(f"Iteration {i//1000}, Mislabeled Postives: {mislabeled_positives}, PU_mlp Accuracy: {pu_mlp_f1score}")
