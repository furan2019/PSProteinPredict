### STEP1-2 ###
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import pandas as pd

Nops_list = pd.read_csv("nops_list.csv", header=None)
Ps_list = pd.read_csv("speci8_list.csv", header=None)

Nops = []
Pas = []
Nops_all = pd.DataFrame()
Pas_all = pd.DataFrame()

for i in range(0,len(Ps_list)):
    Pas.append(pd.read_csv(str((Ps_list.iloc[i]).values[0]), index_col=0))
    Nops.append((pd.read_csv(str((Nops_list.iloc[i]).values[0]), index_col=0)))

for i in Nops:
    Nops_all = pd.concat([Nops_all,i])

for i in Pas:
    Pas_all = pd.concat([Pas_all, i])

params = {
    'objective':'binary',
    'boosting_type':'dart'
}

Nops_all.to_csv("Nops_all.csv")
Pas_all.to_csv("Pas_all.csv")

#
mean_values_ps = Pas_all.mean()
Pas_all_filled = Pas_all.fillna(mean_values_ps)

mean_values_no = Nops_all.mean()
Nops_all_filled = Nops_all.fillna(mean_values_no)

#
verified_positive = pd.read_csv("CorrectedData_positive_755.csv", index_col=0)
mean_values_verified_positive = verified_positive.mean()
verified_positive_filled = verified_positive.fillna(mean_values_verified_positive)

verified_samples = verified_positive_filled.sample(n=700, random_state=0)
verified_remaining = verified_positive_filled.drop(verified_samples.index)
verified_remaining.to_csv("verified_test.csv")
verified_samples.to_csv("verified_samples.csv")

positive_samples_ = Pas_all_filled.sample(n=6000, random_state=0) #
positive_index = positive_samples_.index
positive_test = Pas_all_filled.drop(labels=positive_index)
positive_test.to_csv("positive_test.csv")
positive_samples_.to_csv("positive_samples_.csv")

#oversampling
verified_oversampled = resample(verified_samples, 
                               replace=True, 
                               n_samples=len(positive_samples_), 
                               random_state=0)

background_samples_ = Nops_all_filled.sample(n=144000, random_state=1)  #
background_index = background_samples_.index
background_test = Nops_all_filled.drop(labels=background_index)
background_test.to_csv("background_test.csv")
background_samples_.to_csv("background_samples_.csv")

#
protein_predictions = {}
for i in background_index:
    protein_predictions[i] = []

#
positive_samples_combined = pd.concat([positive_samples_, verified_oversampled])
positive_samples = np.array(positive_samples_combined.drop("Score", axis=1))
background_samples = np.array(background_samples_.drop("Score", axis=1))

#
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

#
average_protein_predictions = {}
for protein_index, predictions in protein_predictions.items():
    if len(predictions) > 0:
        average_prediction = np.mean(predictions)
    else:
        average_prediction = 0.0
    average_protein_predictions[protein_index] = average_prediction

#
output_file = 'LGB_prediction_score.csv'
with open(output_file, 'w') as f:
    f.write("ProteinIndex,LGBscore\n")
    for protein_index, average_prediction in average_protein_predictions.items():
        f.write(f"{protein_index},{average_prediction}\n")

