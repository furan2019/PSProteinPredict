"""
Integrated Machine Learning Pipeline for Binary Classification
Combining LightGBM and PU Learning approaches for protein classification tasks

This script includes:
1. Individual feature analysis
2. Multi-feature interspecies analysis 
3. Multi-feature bulk analysis
4. PU learning comparison
5. Model performance evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, auc, accuracy_score, 
                           f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.utils import resample
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU


class MLPipeline:
    """Main pipeline class for machine learning experiments"""
    
    def __init__(self):
        self.nops_list = pd.read_csv("nops_list.csv", header=None)
        self.ps_list = pd.read_csv("speci8_list.csv", header=None)
        self.species_names = ['At', 'Br', 'Cr', 'Os', 'Pp', 'Sl', 'Ta', 'Zm']
        self.fold_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
        self.models_list = ["RF", "LGB", "DT", "SVM", "NB", "MLP"]
        
    def load_data(self, random_state=2025):
        """Load and prepare positive and negative datasets"""
        nops = []
        pas = []
        
        for i in range(len(self.ps_list)):
            pas.append(pd.read_csv(str(self.ps_list.iloc[i].values[0]), index_col=0))
            nops.append(
                pd.read_csv(str(self.nops_list.iloc[i].values[0]), index_col=0)
                .sample(n=len(pas[i]), random_state=random_state)
            )
        
        # Combine all data
        nops_all = pd.concat(nops, ignore_index=False)
        pas_all = pd.concat(pas, ignore_index=False)
        
        return nops, pas, nops_all.fillna(0), pas_all.fillna(0)
    
    def plot_roc_curves(self, fpr_list, tpr_list, auc_list, labels, title="ROC Curves"):
        """Plot ROC curves for multiple models/features"""
        plt.figure(figsize=(10, 8))
        
        for fpr, tpr, auc_val, label in zip(fpr_list, tpr_list, auc_list, labels):
            plt.plot(fpr, tpr, label=f'{label}, AUC={auc_val:.2f}', linestyle='--')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def individual_feature_analysis(self, random_state=2025):
        """
        Code 1: Individual feature analysis using LightGBM
        Tests each feature individually for classification performance
        """
        print("=== Individual Feature Analysis ===")
        
        _, _, nops_all, pas_all = self.load_data(random_state)
        lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')
        y_pred_pos_acc = pd.DataFrame()
        
        fpr_list, tpr_list, auc_list, labels = [], [], [], []
        
        for i in range(7):  # First 7 features
            # Split data
            test_df_pos = pas_all.sample(n=800, random_state=random_state)
            train_df_pos = pas_all.drop(labels=test_df_pos.index)
            test_df_neg = nops_all.sample(n=800, random_state=random_state)
            train_df_neg = nops_all.drop(labels=test_df_neg.index)
            
            # Select specific feature and score column
            test_df_pos_f = test_df_pos.iloc[:, [i, 7]]
            train_df_pos_f = train_df_pos.iloc[:, [i, 7]]
            test_df_neg_f = test_df_neg.iloc[:, [i, 7]]
            train_df_neg_f = train_df_neg.iloc[:, [i, 7]]
            
            # Prepare training and testing data
            training_df = pd.concat([train_df_pos_f, train_df_neg_f])
            x_train = np.array(training_df.drop("Score", axis=1))
            y_train = np.array(training_df.Score)
            
            testing_df = pd.concat([test_df_pos_f, test_df_neg_f])
            x_test = np.array(testing_df.drop("Score", axis=1))
            y_test = np.array(testing_df.Score)
            x_test_pos = np.array(test_df_pos_f.drop("Score", axis=1))
            y_test_pos = np.array(test_df_pos_f.Score)
            
            # Train model and evaluate
            lgb_model.fit(x_train, y_train)
            fpr, tpr, _ = roc_curve(y_test, lgb_model.predict_proba(x_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            labels.append(f'Feature_{i+1}')
            
            # Calculate accuracy on positive samples
            y_pred_pos = lgb_model.predict(x_test_pos)
            y_pred_pos_acc[i] = [accuracy_score(y_test_pos, y_pred_pos)]
        
        # Plot results
        self.plot_roc_curves(fpr_list, tpr_list, auc_list, labels, 
                           "Individual Feature Analysis - ROC Curves")
        
        # Save results
        y_pred_pos_acc.columns = pas_all.columns[:7]
        y_pred_pos_acc.to_csv("individual_feature_results.csv")
        
        return y_pred_pos_acc
    
    def interspecies_analysis(self, random_state=2025):
        """
        Code 2: Multi-feature interspecies analysis
        Tests model performance across different species
        """
        print("=== Interspecies Analysis ===")
        
        nops, pas, nops_all, pas_all = self.load_data(random_state)
        lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')
        y_pred_pos_acc = pd.DataFrame()
        
        fpr_list, tpr_list, auc_list, labels = [], [], [], []
        
        for i in range(len(self.ps_list)):
            # Use one species for testing, others for training
            test_df_pos = pas[i]
            train_df_pos = pas_all.drop(labels=test_df_pos.index)
            test_df_neg = nops[i]
            train_df_neg = nops_all.drop(labels=test_df_neg.index)
            
            # Prepare data
            training_df = pd.concat([train_df_pos, train_df_neg])
            x_train = np.array(training_df.drop("Score", axis=1))
            y_train = np.array(training_df.Score)
            
            testing_df = pd.concat([test_df_pos, test_df_neg])
            x_test = np.array(testing_df.drop("Score", axis=1))
            y_test = np.array(testing_df.Score)
            x_test_pos = np.array(test_df_pos.drop("Score", axis=1))
            y_test_pos = np.array(test_df_pos.Score)
            
            # Train and evaluate
            lgb_model.fit(x_train, y_train)
            fpr, tpr, _ = roc_curve(y_test, lgb_model.predict_proba(x_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            labels.append(self.species_names[i])
            
            y_pred_pos = lgb_model.predict(x_test_pos)
            y_pred_pos_acc[i] = [accuracy_score(y_test_pos, y_pred_pos)]
        
        # Plot results
        self.plot_roc_curves(fpr_list, tpr_list, auc_list, labels,
                           "Interspecies Analysis - ROC Curves")
        
        # Save results
        y_pred_pos_acc.columns = self.species_names
        y_pred_pos_acc.to_csv("interspecies_results.csv")
        
        return y_pred_pos_acc
    
    def bulk_analysis(self, n_folds=10):
        """
        Code 3: Multi-feature bulk analysis with cross-validation
        Tests model stability across multiple random splits
        """
        print("=== Bulk Analysis ===")
        
        _, pas, _, pas_all = self.load_data()
        lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')
        y_pred_pos_acc = pd.DataFrame()
        
        fpr_list, tpr_list, auc_list, labels = [], [], [], []
        
        for fold in range(n_folds):
            # Load data with different random state for each fold
            nops, _, nops_all, _ = self.load_data(random_state=fold)
            
            # Sample test and train sets
            test_df_pos = pas_all.sample(n=800, random_state=2024)
            train_df_pos = pas_all.drop(labels=test_df_pos.index)
            test_df_neg = nops_all.sample(n=800, random_state=2024)
            train_df_neg = nops_all.drop(labels=test_df_neg.index)
            
            # Prepare data
            training_df = pd.concat([train_df_pos, train_df_neg])
            x_train = np.array(training_df.drop("Score", axis=1))
            y_train = np.array(training_df.Score)
            
            testing_df = pd.concat([test_df_pos, test_df_neg])
            x_test = np.array(testing_df.drop("Score", axis=1))
            y_test = np.array(testing_df.Score)
            x_test_pos = np.array(test_df_pos.drop("Score", axis=1))
            y_test_pos = np.array(test_df_pos.Score)
            
            # Train and evaluate
            lgb_model.fit(x_train, y_train)
            fpr, tpr, _ = roc_curve(y_test, lgb_model.predict_proba(x_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            labels.append(self.fold_names[fold])
            
            y_pred_pos = lgb_model.predict(x_test_pos)
            y_pred_pos_acc[fold] = [accuracy_score(y_test_pos, y_pred_pos)]
        
        # Plot results
        self.plot_roc_curves(fpr_list, tpr_list, auc_list, labels,
                           "Bulk Analysis - ROC Curves")
        
        # Save results
        y_pred_pos_acc.columns = self.fold_names
        y_pred_pos_acc.to_csv("bulk_analysis_results.csv")
        
        return y_pred_pos_acc
    
    def pu_learning_comparison(self, n_folds=10):
        """
        Code 4 & 5: PU Learning comparison with traditional methods
        Compares standard LightGBM with PU learning approach
        """
        print("=== PU Learning Comparison ===")
        
        _, pas, _, pas_all = self.load_data()
        lgb_model = LGBMClassifier(objective='binary', boosting_type='dart')
        pu_model = BaggingClassifierPU(base_estimator=lgb_model)
        
        y_pred_pos_acc_lgb = pd.DataFrame()
        y_pred_pos_acc_pu = pd.DataFrame()
        
        fpr_list, tpr_list, auc_list, labels = [], [], [], []
        
        for fold in range(n_folds):
            # Load data with different random state
            nops, _, nops_all, _ = self.load_data(random_state=fold)
            
            # Sample data
            test_df_pos = pas_all.sample(n=800, random_state=2025)
            train_df_pos = pas_all.drop(labels=test_df_pos.index)
            test_df_neg = nops_all.sample(n=800, random_state=2025)
            train_df_neg = nops_all.drop(labels=test_df_neg.index)
            
            # Prepare data
            training_df = pd.concat([train_df_pos, train_df_neg])
            x_train = np.array(training_df.drop("Score", axis=1))
            y_train = np.array(training_df.Score)
            
            testing_df = pd.concat([test_df_pos, test_df_neg])
            x_test = np.array(testing_df.drop("Score", axis=1))
            y_test = np.array(testing_df.Score)
            x_test_pos = np.array(test_df_pos.drop("Score", axis=1))
            y_test_pos = np.array(test_df_pos.Score)
            
            # Train standard LightGBM
            lgb_model.fit(x_train, y_train)
            y_pred_pos_lgb = lgb_model.predict(x_test_pos)
            y_pred_pos_acc_lgb[fold] = [accuracy_score(y_test_pos, y_pred_pos_lgb)]
            
            # Train PU model
            pu_model.fit(x_train, y_train)
            y_pred_pos_pu = pu_model.predict(x_test_pos)
            y_pred_pos_acc_pu[fold] = [accuracy_score(y_test_pos, y_pred_pos_pu)]
            
            # ROC curve for PU model
            fpr, tpr, _ = roc_curve(y_test, pu_model.predict_proba(x_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            labels.append(self.fold_names[fold])
        
        # Plot PU learning ROC curves
        self.plot_roc_curves(fpr_list, tpr_list, auc_list, labels,
                           "PU Learning - ROC Curves")
        
        # Combine results
        y_pred_pos_acc = pd.concat([y_pred_pos_acc_lgb, y_pred_pos_acc_pu])
        y_pred_pos_acc.columns = self.fold_names
        y_pred_pos_acc.index = ['LGB', 'PU']
        y_pred_pos_acc.to_csv("pu_learning_comparison.csv")
        
        return y_pred_pos_acc
    
    def get_classifier(self, model_name):
        """Get classifier instance by name"""
        if model_name == "SVM":
            from sklearn.svm import SVC
            return SVC(kernel='rbf', gamma='auto', random_state=0)
        elif model_name == "DT":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier()
        elif model_name == "RF":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=10, random_state=0)
        elif model_name == "LGB":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(objective='binary', boosting_type='dart')
        elif model_name == "NB":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        elif model_name == "MLP":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam')
    
    def random_undersampling(self, df, target_label):
        """Perform random undersampling to balance classes"""
        df_majority = df[df[target_label] == 0]
        df_minority = df[df[target_label] == 1]
        
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=None
        )
        
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        print(f"Undersampling complete! Class distribution: {df_downsampled[target_label].value_counts().to_dict()}")
        
        return df_downsampled
    
    def comprehensive_model_evaluation(self, n_folds=10, sample_num=800, hidden_size=400):
        """
        Code 6: Comprehensive model evaluation with multiple algorithms
        Tests various classifiers with and without PU learning
        """
        print("=== Comprehensive Model Evaluation ===")
        
        _, pas, _, pas_all = self.load_data()
        
        # Initialize result dataframes
        metrics = ['accuracy', 'recall', 'f1_score']
        results = {}
        for metric in metrics:
            results[f'{metric}_standard'] = pd.DataFrame(index=self.models_list)
            results[f'{metric}_pu'] = pd.DataFrame(index=self.models_list)
        
        for fold in range(n_folds):
            print(f"Processing fold {fold + 1}/{n_folds}")
            
            # Load data with different random state
            nops, _, nops_all, _ = self.load_data(random_state=fold)
            
            # Sample and prepare data
            test_df_pos_raw = pas_all.sample(n=sample_num, random_state=fold)
            test_df_neg_raw = nops_all.sample(n=sample_num, random_state=fold)
            df_raw = pd.concat([test_df_pos_raw, test_df_neg_raw])
            
            # Balance classes
            df_downsampled = self.random_undersampling(df_raw, 'Score')
            df_downsampled = df_downsampled.sample(frac=1).reset_index(drop=True)
            
            # Remove non-feature columns if they exist
            if 'UniprotEntry' in df_downsampled.columns:
                df_downsampled = df_downsampled.drop(columns=['UniprotEntry'])
            
            # Prepare features and labels
            X = df_downsampled.drop('Score', axis=1)
            y_orig = df_downsampled['Score'].copy()
            
            # Create hidden labels (simulate unlabeled positive samples)
            y_hidden = y_orig.copy()
            positive_indices = y_hidden[y_hidden == 1].index
            indices_to_hide = np.random.choice(positive_indices, replace=False, size=hidden_size)
            y_hidden.loc[indices_to_hide] = 0
            
            # Evaluate standard models
            fold_results = {metric: [] for metric in metrics}
            for model_name in self.models_list:
                classifier = self.get_classifier(model_name)
                classifier.fit(X, y_hidden)
                
                y_pred = classifier.predict(X)
                fold_results['accuracy'].append(accuracy_score(y_orig, y_pred))
                fold_results['recall'].append(recall_score(y_orig, y_pred))
                fold_results['f1_score'].append(f1_score(y_orig, y_pred))
            
            # Store standard results
            for metric in metrics:
                results[f'{metric}_standard'][fold] = fold_results[metric]
            
            # Evaluate PU learning models
            fold_results_pu = {metric: [] for metric in metrics}
            for model_name in self.models_list:
                base_classifier = self.get_classifier(model_name)
                pu_classifier = BaggingClassifierPU(
                    base_classifier,
                    n_estimators=100,
                    max_samples=sum(y_hidden),
                    bootstrap=True,
                    oob_score=True,
                    n_jobs=-1
                )
                pu_classifier.fit(X, y_hidden)
                
                y_pred_pu = pu_classifier.predict(X)
                fold_results_pu['accuracy'].append(accuracy_score(y_orig, y_pred_pu))
                fold_results_pu['recall'].append(recall_score(y_orig, y_pred_pu))
                fold_results_pu['f1_score'].append(f1_score(y_orig, y_pred_pu))
            
            # Store PU results
            for metric in metrics:
                results[f'{metric}_pu'][fold] = fold_results_pu[metric]
        
        # Calculate means and save results
        for metric in metrics:
            standard_mean = results[f'{metric}_standard'].mean(axis=1)
            pu_mean = results[f'{metric}_pu'].mean(axis=1)
            
            comparison_df = pd.DataFrame({
                'standard': standard_mean,
                'pu_learning': pu_mean
            }, index=self.models_list)
            
            comparison_df.to_csv(f"{metric}_comparison_results.csv")
            print(f"\n{metric.upper()} Results:")
            print(comparison_df)
        
        return results
    
    def run_full_pipeline(self):
        """Run the complete analysis pipeline"""
        print("Starting Full ML Pipeline Analysis")
        print("=" * 50)
        
        try:
            # Run all analyses
            individual_results = self.individual_feature_analysis()
            interspecies_results = self.interspecies_analysis()
            bulk_results = self.bulk_analysis()
            pu_comparison = self.pu_learning_comparison()
            comprehensive_results = self.comprehensive_model_evaluation()
            
            print("\n" + "=" * 50)
            print("Pipeline completed successfully!")
            print("Results saved to CSV files:")
            print("- individual_feature_results.csv")
            print("- interspecies_results.csv") 
            print("- bulk_analysis_results.csv")
            print("- pu_learning_comparison.csv")
            print("- accuracy_comparison_results.csv")
            print("- recall_comparison_results.csv")
            print("- f1_score_comparison_results.csv")
            
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Run individual analyses or full pipeline
    print("Choose analysis to run:")
    print("1. Individual Feature Analysis")
    print("2. Interspecies Analysis")
    print("3. Bulk Analysis")
    print("4. PU Learning Comparison")
    print("5. Comprehensive Model Evaluation")
    print("6. Full Pipeline")
    
    choice = input("Enter choice (1-6): ").strip()
    
    try:
        if choice == '1':
            pipeline.individual_feature_analysis()
        elif choice == '2':
            pipeline.interspecies_analysis()
        elif choice == '3':
            pipeline.bulk_analysis()
        elif choice == '4':
            pipeline.pu_learning_comparison()
        elif choice == '5':
            pipeline.comprehensive_model_evaluation()
        elif choice == '6':
            pipeline.run_full_pipeline()
        else:
            print("Invalid choice. Running full pipeline by default.")
            pipeline.run_full_pipeline()
            
    except FileNotFoundError as e:
        print(f"Required data files not found: {e}")
        print("Please ensure the following files are in the current directory:")
        print("- nops_list.csv")
        print("- speci8_list.csv")
        print("- Individual species data files listed in the CSV files")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
