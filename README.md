# PSProteinPredict_V1.0
A comprehensive machine learning framework for plant liquid-liquid phase separation (LLPS) protein prediction using LightGBM and Positive-Unlabeled (PU) learning approaches.

## Overview

PSProteinPredict V1.0 is a computational tool designed for predicting plant phase separation proteins through advanced machine learning techniques. The framework combines LightGBM gradient boosting with PU learning to address the fundamental challenge in LLPS protein prediction: identifying positive samples from unlabeled protein databases where explicit negative samples are unavailable.

## Repository Structure

PSProteinPredict_V1.0/

├── Figure1-2_LGB_PU_code1-6.py          # LightGBM + PU learning binary classification

├── Figure3_PPF_code_gather.py           # Feature combination evaluation

├── Figure4_wholePro_code_gather.py      # Whole proteome prediction

├── Figure1-2_LGB_PU/                    # the streamlined code and corresponding data for every panel of Figures 1 to Figure 2

├── Figure3_PPF/                         # the streamlined code and corresponding data for every panel of Figures 3

├── Figure3_RSA_PTS/                     # the streamlined code and corresponding data for every panel of Figures 3

├── Figure4_wholePro/                    # the streamlined code and corresponding data for every panel of Figures 4

├── Figure5_recall_validation/           # the streamlined code and corresponding data for every panel of Figures 5

└── README.md                            # This file

## Main Scripts

### 1.Figure1-2_LGB_PU_code1-6.py
   
#### Purpose: 

Integrated Machine Learning Pipeline for Binary Classification
Combining LightGBM and PU Learning approaches for protein classification tasks

#### This script includes:

(1) Individual feature analysis
(2) Multi-feature interspecies analysis 
(3) Multi-feature bulk analysis
(4) PU learning comparison
(5) Model performance evaluation

### 2.Figure3_PPF_code_gather.py

#### Purpose: 
Protein phosphorylation site prediction with feature combination analysis.
PPF (Phosphorylation Frequency) Analysis Pipeline.

#### This integrated script combines four analysis approaches:

(1) PPF Known Data Analysis (Figure 3c)
(2) PPF Known + Predicted Data Analysis (Figure 3c)  
(3) PPF Predicted Data Analysis (Figure 3c)
(4) Recall Analysis with 7 Features + PPF (Figure 3d)

##### Features analyzed:
F1: PhosFreq only
F7: 7 sequence-based feature
F8: 7 original features + PhosFreq

### 3.Figure4_wholePro_code_gather.py

#### Purpose: 
Integrated Protein Classification PU Learning Pipeline
Complete workflow including data preprocessing, model training, scoring and reliability classification

Figure-Specific Directories
### Each Figure*/ directory contains:

#### Simplified code versions: Streamlined scripts for specific analyses

#### Dataset files: Curated data used for each figure's experiments

#### Configuration files: Parameter settings and experimental configurations

#### Results: Output files and performance metrics

