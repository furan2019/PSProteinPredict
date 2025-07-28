"""
PPF (Phosphorylation Frequency) Analysis Pipeline
Feature Comparison Study for Protein Phosphorylation Site Prediction

This integrated script combines four analysis approaches:
1. PPF Known Data Analysis (Figure 3c)
2. PPF Known + Predicted Data Analysis (Figure 3c)  
3. PPF Predicted Data Analysis (Figure 3c)
4. Recall Analysis with 7 Features + PPF (Figure 3d)

Features analyzed:
- F1: PhosFreq only
- F7: 7 original features (excluding PhosFreq)
- F8: 7 original features + PhosFreq
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU
import warnings
warnings.filterwarnings('ignore')


class PPFAnalysisPipeline:
    """Main pipeline class for PPF feature analysis"""
    
    def __init__(self, sample_size=100, n_outer_folds=10, n_inner_folds=10):
        """
        Initialize the pipeline
        
        Args:
            sample_size (int): Sample size for testing (default: 100)
            n_outer_folds (int): Number of outer cross-validation folds (default: 10)
            n_inner_folds (int): Number of inner cross-validation folds (default: 10)
        """
        self.sample_size = sample_size
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        
        # Feature configurations
        self.feature_configs = {
            'F1': 'PhosFreq only',
            'F7': '7 features (without PhosFreq)', 
            'F8': '7 features + PhosFreq'
        }
        
        # Model configurations
        self.model_configs = {
            'lgb': 'LightGBM',
            'pu': 'PU Learning + LightGBM'
        }
        
    def get_classifier(self, model_type='lgb'):
        """Get classifier instance"""
        base_classifier = LGBMClassifier(objective='binary', boosting_type='dart', verbose=-1)
        
        if model_type == 'lgb':
            return base_classifier
        elif model_type == 'pu':
            return BaggingClassifierPU(base_estimator=base_classifier)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_and_evaluate(self, X_train, y_train, X_test, y_test, model_type='lgb'):
        """
        Train model and evaluate performance
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data  
            model_type: 'lgb' or 'pu'
            
        Returns:
            accuracy, recall, f1_score
        """
        classifier = self.get_classifier(model_type)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return accuracy, recall, f1
    
    def prepare_feature_data(self, pos_df, neg_df, feature_type):
        """
        Prepare data based on feature configuration
        
        Args:
            pos_df, neg_df: Positive and negative dataframes
            feature_type: 'F1', 'F7', or 'F8'
            
        Returns:
            Modified dataframes based on feature selection
        """
        if feature_type == 'F1':  # PhosFreq only
            pos_df_feat = pos_df.iloc[:, 7:9]  # PhosFreq + Score columns
            neg_df_feat = neg_df.iloc[:, 7:9]
        elif feature_type == 'F7':  # 7 features without PhosFreq
            pos_df_feat = pos_df.drop("PhosFreq", axis=1)
            neg_df_feat = neg_df.drop("PhosFreq", axis=1)
        elif feature_type == 'F8':  # All 8 features
            pos_df_feat = pos_df.copy()
            neg_df_feat = neg_df.copy()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        return pos_df_feat, neg_df_feat
    
    def load_and_process_data(self, data_type='known'):
        """
        Load and process data based on data type
        
        Args:
            data_type: 'known', 'predicted', or 'combined'
            
        Returns:
            pos_df, neg_df: Processed positive and negative dataframes
        """
        if data_type == 'known':
            # Load known data only
            pos_df = pd.read_csv("PS_7features+PPF.csv", index_col=0)
            neg_df = pd.read_csv("NonPS_7features+PPF.csv", index_col=0)
            
        elif data_type == 'predicted':
            # Load predicted data only
            pos_df = pd.read_csv("PS_7features+PPF_predicted.csv", index_col=0)
            neg_df = pd.read_csv("NonPS_7features+PPF_predicted.csv", index_col=0)
            
        elif data_type == 'combined':
            # Load and combine known + predicted data
            pos_known = pd.read_csv("PS_7features+PPF.csv", index_col=0)
            neg_known = pd.read_csv("NonPS_7features+PPF.csv", index_col=0)
            pos_pred = pd.read_csv("PS_7features+PPF_predicted.csv", index_col=0)
            neg_pred = pd.read_csv("NonPS_7features+PPF_predicted.csv", index_col=0)
            
            pos_df = pd.concat([pos_known, pos_pred])
            neg_df = pd.concat([neg_known, neg_pred])
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Fill missing values with mean
        mean_values_pos = pos_df.mean()
        pos_df_filled = pos_df.fillna(mean_values_pos)
        mean_values_neg = neg_df.mean()
        neg_df_filled = neg_df.fillna(mean_values_neg)
        
        return pos_df_filled, neg_df_filled
    
    def run_feature_comparison_analysis(self, data_type='known'):
        """
        Run feature comparison analysis (Figure 3c equivalent)
        
        Args:
            data_type: 'known', 'predicted', or 'combined'
            
        Returns:
            results_df: DataFrame with TPR and F1-score results
        """
        print(f"=== Feature Comparison Analysis - {data_type.upper()} Data ===")
        
        pos_df, neg_df = self.load_and_process_data(data_type)
        
        # Initialize result storage
        results = {
            'F1_lgb': {'accuracy': [], 'f1': []},
            'F7_lgb': {'accuracy': [], 'f1': []},
            'F8_lgb': {'accuracy': [], 'f1': []},
            'F8_pu': {'accuracy': [], 'f1': []}
        }
        
        for outer_fold in range(self.n_outer_folds):
            print(f"Processing outer fold {outer_fold + 1}/{self.n_outer_folds}")
            
            # Sample negative data to match positive data size
            neg_df_sampled = neg_df.sample(n=len(pos_df), random_state=outer_fold)
            
            # Temporary storage for inner folds
            fold_results = {key: {'accuracy': [], 'f1': []} for key in results.keys()}
            
            for inner_fold in range(self.n_inner_folds):
                # Sample test data
                test_pos = pos_df.sample(n=self.sample_size, random_state=inner_fold)
                train_pos = pos_df.drop(labels=test_pos.index)
                test_neg = neg_df_sampled.sample(n=self.sample_size, random_state=inner_fold)
                train_neg = neg_df_sampled.drop(labels=test_neg.index)
                
                # Test different feature configurations
                for feature_type in ['F1', 'F7', 'F8']:
                    # Prepare feature data
                    test_pos_feat, test_neg_feat = self.prepare_feature_data(test_pos, test_neg, feature_type)
                    train_pos_feat, train_neg_feat = self.prepare_feature_data(train_pos, train_neg, feature_type)
                    
                    # Combine training data
                    train_df = pd.concat([train_pos_feat, train_neg_feat])
                    X_train = np.array(train_df.drop("Score", axis=1))
                    y_train = np.array(train_df.Score)
                    
                    # Prepare test data (positive only for TPR calculation)
                    X_test_pos = np.array(test_pos_feat.drop("Score", axis=1))
                    y_test_pos = np.array(test_pos_feat.Score)
                    
                    # Prepare combined test data (for F1 calculation)
                    test_df = pd.concat([test_pos_feat, test_neg_feat])
                    X_test_all = np.array(test_df.drop("Score", axis=1))
                    y_test_all = np.array(test_df.Score)
                    
                    # Test LightGBM
                    accuracy_pos, _, _ = self.predict_and_evaluate(X_train, y_train, X_test_pos, y_test_pos, 'lgb')
                    _, _, f1_all = self.predict_and_evaluate(X_train, y_train, X_test_all, y_test_all, 'lgb')
                    
                    fold_results[f'{feature_type}_lgb']['accuracy'].append(accuracy_pos)
                    fold_results[f'{feature_type}_lgb']['f1'].append(f1_all)
                    
                    # Test PU Learning (only for F8)
                    if feature_type == 'F8':
                        accuracy_pos_pu, _, _ = self.predict_and_evaluate(X_train, y_train, X_test_pos, y_test_pos, 'pu')
                        _, _, f1_all_pu = self.predict_and_evaluate(X_train, y_train, X_test_all, y_test_all, 'pu')
                        
                        fold_results['F8_pu']['accuracy'].append(accuracy_pos_pu)
                        fold_results['F8_pu']['f1'].append(f1_all_pu)
            
            # Calculate means for this outer fold
            for config in results.keys():
                results[config]['accuracy'].append(np.mean(fold_results[config]['accuracy']))
                results[config]['f1'].append(np.mean(fold_results[config]['f1']))
        
        # Calculate final means
        final_results = {}
        for config in results.keys():
            final_results[config] = {
                'TPR': np.mean(results[config]['accuracy']),
                'F1score': np.mean(results[config]['f1'])
            }
        
        # Create results DataFrame
        results_df = pd.DataFrame(final_results).T
        results_df = results_df[['TPR', 'F1score']]  # Ensure column order
        
        # Save results
        filename = f"PPF_{data_type}_Figure3c_results.csv"
        results_df.to_csv(filename)
        print(f"Results saved to {filename}")
        
        return results_df
    
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
    
    def run_recall_analysis(self, sample_num=400, hidden_size=200):
        """
        Run recall analysis with hidden positive labels (Figure 3d equivalent)
        
        Args:
            sample_num: Sample size for analysis
            hidden_size: Number of positive samples to hide
            
        Returns:
            results_df: DataFrame with recall, F1-score, and accuracy results
        """
        print("=== Recall Analysis with Hidden Labels ===")
        
        # Load combined data (known + predicted)
        pos_df, neg_df = self.load_and_process_data('combined')
        
        # Initialize result storage
        results_lgb = {'accuracy': [], 'recall': [], 'f1_score': []}
        results_pu = {'accuracy': [], 'recall': [], 'f1_score': []}
        
        for fold in range(self.n_outer_folds):
            print(f"Processing fold {fold + 1}/{self.n_outer_folds}")
            
            # Sample negative data to match positive data size
            neg_df_sampled = neg_df.sample(n=len(pos_df), random_state=fold)
            
            # Sample raw test data
            test_pos_raw = pos_df.sample(n=sample_num, random_state=fold)
            test_neg_raw = neg_df_sampled.sample(n=sample_num, random_state=fold)
            df_raw = pd.concat([test_pos_raw, test_neg_raw])
            
            print(f"Original class distribution: {df_raw.Score.value_counts().to_dict()}")
            
            # Balance classes through undersampling
            df_balanced = self.random_undersampling(df_raw, 'Score')
            df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
            
            # Remove non-feature columns if they exist
            columns_to_drop = ['UniproEntry', 'UniprotEntry']
            for col in columns_to_drop:
                if col in df_balanced.columns:
                    df_balanced = df_balanced.drop(columns=[col])
            
            # Prepare features and labels
            X = df_balanced.drop('Score', axis=1)
            y_orig = df_balanced['Score'].copy()
            
            # Create hidden labels (simulate unlabeled positive samples)
            y_hidden = y_orig.copy()
            positive_indices = y_hidden[y_hidden == 1].index
            
            # Ensure we don't try to hide more samples than available
            actual_hidden_size = min(hidden_size, len(positive_indices))
            if actual_hidden_size > 0:
                indices_to_hide = np.random.choice(positive_indices, replace=False, size=actual_hidden_size)
                y_hidden.loc[indices_to_hide] = 0
                print(f"Hidden {actual_hidden_size} positive samples")
            
            print(f"Hidden class distribution: {y_hidden.value_counts().to_dict()}")
            
            # Evaluate LightGBM
            accuracy_lgb, recall_lgb, f1_lgb = self.predict_and_evaluate(X, y_hidden, X, y_orig, 'lgb')
            results_lgb['accuracy'].append(accuracy_lgb)
            results_lgb['recall'].append(recall_lgb)
            results_lgb['f1_score'].append(f1_lgb)
            
            # Evaluate PU Learning
            accuracy_pu, recall_pu, f1_pu = self.predict_and_evaluate(X, y_hidden, X, y_orig, 'pu')
            results_pu['accuracy'].append(accuracy_pu)
            results_pu['recall'].append(recall_pu)
            results_pu['f1_score'].append(f1_pu)
        
        # Calculate means
        lgb_means = [np.mean(results_lgb['recall']), np.mean(results_lgb['f1_score']), np.mean(results_lgb['accuracy'])]
        pu_means = [np.mean(results_pu['recall']), np.mean(results_pu['f1_score']), np.mean(results_pu['accuracy'])]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'lgb': lgb_means,
            'pu': pu_means
        }, index=['Recall', 'F1_score', 'Accuracy'])
        
        # Save results
        filename = "Recall_7features+PPF_Figure3d_results.csv"
        results_df.to_csv(filename)
        print(f"Results saved to {filename}")
        
        return results_df
    
    def plot_comparison_results(self, results_dict, title="Feature Comparison Results"):
        """
        Plot comparison results
        
        Args:
            results_dict: Dictionary of results DataFrames
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot TPR comparison
        tpr_data = {}
        f1_data = {}
        
        for data_type, df in results_dict.items():
            tpr_data[data_type] = df['TPR'].values
            f1_data[data_type] = df['F1score'].values
        
        # TPR plot
        x = np.arange(len(df.index))
        width = 0.25
        
        for i, (data_type, values) in enumerate(tpr_data.items()):
            ax1.bar(x + i*width, values, width, label=data_type, alpha=0.8)
        
        ax1.set_xlabel('Feature Configuration')
        ax1.set_ylabel('True Positive Rate (TPR)')
        ax1.set_title('TPR Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(df.index)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1-score plot
        for i, (data_type, values) in enumerate(f1_data.items()):
            ax2.bar(x + i*width, values, width, label=data_type, alpha=0.8)
        
        ax2.set_xlabel('Feature Configuration')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(df.index)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis including all data types and visualizations"""
        print("Starting Comprehensive PPF Analysis Pipeline")
        print("=" * 60)
        
        try:
            # Run feature comparison analyses
            print("\n1. Running Feature Comparison Analyses...")
            results_known = self.run_feature_comparison_analysis('known')
            print("\nKnown Data Results:")
            print(results_known)
            
            results_predicted = self.run_feature_comparison_analysis('predicted')
            print("\nPredicted Data Results:")
            print(results_predicted)
            
            results_combined = self.run_feature_comparison_analysis('combined')
            print("\nCombined Data Results:")
            print(results_combined)
            
            # Plot feature comparison results
            comparison_results = {
                'Known': results_known,
                'Predicted': results_predicted,
                'Combined': results_combined
            }
            self.plot_comparison_results(comparison_results, "Feature Comparison Across Data Types")
            
            # Run recall analysis
            print("\n2. Running Recall Analysis with Hidden Labels...")
            recall_results = self.run_recall_analysis()
            print("\nRecall Analysis Results:")
            print(recall_results)
            
            # Plot recall results
            self.plot_recall_results(recall_results)
            
            print("\n" + "=" * 60)
            print("Comprehensive Analysis Completed Successfully!")
            print("\nFiles generated:")
            print("- PPF_known_Figure3c_results.csv")
            print("- PPF_predicted_Figure3c_results.csv")
            print("- PPF_combined_Figure3c_results.csv")
            print("- Recall_7features+PPF_Figure3d_results.csv")
            
            return {
                'feature_comparison': comparison_results,
                'recall_analysis': recall_results
            }
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    def plot_recall_results(self, results_df):
        """Plot recall analysis results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df.index))
        width = 0.35
        
        lgb_values = results_df['lgb'].values
        pu_values = results_df['pu'].values
        
        ax.bar(x - width/2, lgb_values, width, label='LightGBM', alpha=0.8)
        ax.bar(x + width/2, pu_values, width, label='PU Learning', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Recall Analysis: LightGBM vs PU Learning')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df.index)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (lgb_val, pu_val) in enumerate(zip(lgb_values, pu_values)):
            ax.text(i - width/2, lgb_val + 0.01, f'{lgb_val:.3f}', ha='center', va='bottom')
            ax.text(i + width/2, pu_val + 0.01, f'{pu_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = PPFAnalysisPipeline(sample_size=100, n_outer_folds=10, n_inner_folds=10)
    
    print("PPF Analysis Pipeline")
    print("Choose analysis to run:")
    print("1. Feature Comparison - Known Data")
    print("2. Feature Comparison - Predicted Data") 
    print("3. Feature Comparison - Combined Data")
    print("4. Recall Analysis with Hidden Labels")
    print("5. Comprehensive Analysis (All)")
    
    choice = input("Enter choice (1-5): ").strip()
    
    try:
        if choice == '1':
            results = pipeline.run_feature_comparison_analysis('known')
            print("\nResults:")
            print(results)
            
        elif choice == '2':
            results = pipeline.run_feature_comparison_analysis('predicted')
            print("\nResults:")
            print(results)
            
        elif choice == '3':
            results = pipeline.run_feature_comparison_analysis('combined')
            print("\nResults:")
            print(results)
            
        elif choice == '4':
            results = pipeline.run_recall_analysis()
            print("\nResults:")
            print(results)
            
        elif choice == '5':
            pipeline.run_comprehensive_analysis()
            
        else:
            print("Invalid choice. Running comprehensive analysis by default.")
            pipeline.run_comprehensive_analysis()
            
    except FileNotFoundError as e:
        print(f"Required data files not found: {e}")
        print("Please ensure the following files are in the current directory:")
        print("- PS_7features+PPF.csv")
        print("- NonPS_7features+PPF.csv")
        print("- PS_7features+PPF_predicted.csv")
        print("- NonPS_7features+PPF_predicted.csv")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
