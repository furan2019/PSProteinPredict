import numpy as np
from lightgbm import LGBMClassifier
import pandas as pd
import os
from baggingPU import BaggingClassifierPU
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample


#positive training set# 2100
verified_positive_samples = pd.read_csv("verified_samples_positive.csv", index_col=0) #700
plant_positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0) #6000
plant_positive_samples_700 = plant_positive_samples_.sample(n=len(verified_positive_samples), random_state=2025)
positive_samples_combined_ = pd.concat([plant_positive_samples_700, verified_positive_samples]) #1400

predicted_positive_samples_ = pd.read_csv("unlabel_2_ps_new.csv", index_col=0)
mean_predicted_positive_samples_ = predicted_positive_samples_.mean()
predicted_positive_samples_fill = predicted_positive_samples_.fillna(mean_predicted_positive_samples_)
predicted_positive_samples_700 = predicted_positive_samples_fill.sample(n=len(verified_positive_samples), random_state=2025)

positive_samples_combined = pd.concat([predicted_positive_samples_700, positive_samples_combined_]) #2100个


#negative training set# 2100
negative_valid_ = pd.read_csv("CorrectedData_negative_2121.csv", index_col=0)
mean_negative_valid = negative_valid_.mean()
negative_valid = negative_valid_.fillna(mean_negative_valid)
verified_negative_samples = negative_valid.sample(n=1400, random_state=2024)

predicted_negative_samples_ = pd.read_csv("unlabel_2_nops_new.csv", index_col=0)
mean_predicted_negative_samples_ = predicted_negative_samples_.mean()
predicted_negative_samples_fill = predicted_negative_samples_.fillna(mean_predicted_negative_samples_)
predicted_negative_samples_700 = predicted_negative_samples_fill.sample(n=700, random_state=2025)

negative_samples_combined = pd.concat([predicted_negative_samples_700, verified_negative_samples]) #2100个

#
train_data = pd.concat([positive_samples_combined, negative_samples_combined])

#positive validation set# 217
verified_positive_remaining = pd.read_csv("verified_test_positive.csv", index_col=0) #55
plant_positive_valid = pd.read_csv("positive_test.csv", index_col=0)
plant_positive_valid_55 = plant_positive_valid.sample(n=70, random_state=2025)
positive_valid_combined_ = pd.concat([plant_positive_valid_55, verified_positive_remaining]) 
#
published_plant_ = pd.read_csv("published_plant_92.csv", index_col=0)
mean_published_plant = published_plant_.mean()
published_plant = published_plant_.fillna(mean_published_plant)

positive_valid_combined = pd.concat([published_plant, positive_valid_combined_]) #217个

#negative validation set#
negative_valid_217 = negative_valid.sample(n=217, random_state=2025) 
valid_data = pd.concat([positive_valid_combined, negative_valid_217])
#
base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', learning_rate=0.5,
                                            bagging_fraction=0.5,
                                            feature_fraction=0.8, min_child_samples=18, num_leaves=5, random_state=2)

model = BaggingClassifierPU(base_estimator, random_state=2)
model.fit(train_data.drop('Score', axis=1), train_data['Score'])

model_filename = 'pu_bagging_model.pkl'
joblib.dump(model, model_filename)

# model = joblib.load(model_filename)

valid_proba = model.predict_proba(valid_data.drop('Score', axis=1))
valid_predictions = model.predict(valid_data.drop('Score', axis=1))

#save
valid_results = valid_data.copy()
valid_results['Predicted_Probability'] = valid_proba[:, 1]
valid_results['Predicted_Label'] = valid_predictions

valid_results_filename = 'validation_predictions.csv'
valid_results.to_csv(valid_results_filename)
