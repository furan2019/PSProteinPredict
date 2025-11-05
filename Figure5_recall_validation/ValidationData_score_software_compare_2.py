import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score

#
df = pd.read_csv("ValidationData_score_software_compare.csv")
y_true = df.iloc[:, 1]  #

#
software_columns = df.columns[2:8].tolist()
software_names = software_columns
software_scores = {name: df[name] for name in software_names}

print(f"\nlabel column: {df.columns[1]}")
print(f"software column: {software_names}")

#
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

#
def evaluate_software(name, scores, y_true):
    """Evaluate the performance of individual software programs and handle missing values."""
    #
    non_null_mask = ~scores.isnull()
    
    if non_null_mask.sum() == 0:
        print(f"{name}: No valid data")
        return None
    
    y_true_subset = y_true[non_null_mask]
    scores_subset = scores[non_null_mask]
    
    print(f"{name}: Valid sample size = {len(scores_subset)}")
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_true_subset, scores_subset)
    roc_auc = auc(fpr, tpr)
    
    # Calculate the F1 score (using 0.5 as the threshold).
    threshold = 0.5
    y_pred = (scores_subset >= threshold).astype(int)
    f1 = f1_score(y_true_subset, y_pred)
    
    # Calculating positive sample accuracy (TPR)
    positive_mask = y_true_subset == 1
    if sum(positive_mask) > 0:
        pos_acc = accuracy_score(y_true_subset[positive_mask], y_pred[positive_mask])
    else:
        pos_acc = 0
    
    # Calculate other indicators
    precision = precision_score(y_true_subset, y_pred)
    recall = recall_score(y_true_subset, y_pred)
    overall_accuracy = accuracy_score(y_true_subset, y_pred)
    
    return {
        'name': name,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'positive_accuracy': pos_acc,
        'precision': precision,
        'recall': recall,
        'overall_accuracy': overall_accuracy,
        'sample_count': len(scores_subset)
    }

# 1. Plot the ROC curve and calculate the AUC value.
plt.figure(figsize=(10, 8))
performance_results = {}

for name, scores in software_scores.items():
    result = evaluate_software(name, scores, y_true)
    if result is not None:
        performance_results[name] = result
        plt.plot(result['fpr'], result['tpr'], 
                label=f"{name} (AUC = {result['roc_auc']:.3f}, N={result['sample_count']})", 
                linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC curves from various software programs')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ROC_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Calculate the F1 score and plot a bar chart.
plt.figure(figsize=(10, 6))
f1_values = [result['f1_score'] for result in performance_results.values()]
software_names_plot = [result['name'] for result in performance_results.values()]

colors = plt.cm.Set3(np.linspace(0, 1, len(f1_values)))
bars = plt.bar(range(len(f1_values)), f1_values, color=colors)
plt.xticks(range(len(f1_values)), software_names_plot, rotation=45)
plt.ylabel('F1 score')
plt.title('Comparison of F1 scores across different software programs (threshold=0.5)')
plt.ylim(0, 1)

#
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('F1_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Calculate the prediction accuracy for positive samples and plot a bar chart.
plt.figure(figsize=(10, 6))
positive_acc_values = [result['positive_accuracy'] for result in performance_results.values()]

colors = plt.cm.Pastel1(np.linspace(0, 1, len(positive_acc_values)))
bars = plt.bar(range(len(positive_acc_values)), positive_acc_values, color=colors)
plt.xticks(range(len(positive_acc_values)), software_names_plot, rotation=45)
plt.ylabel('TPR')
plt.title('Comparison of positive sample prediction accuracy among various software programs (threshold=0.5)')
plt.ylim(0, 1)

# 
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('positive_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Output detailed performance index tables
performance_table = pd.DataFrame([{
    'Software': result['name'],
    'AUC': result['roc_auc'],
    'F1_Score': result['f1_score'],
    'Positive_Accuracy': result['positive_accuracy'],
    'Precision': result['precision'],
    'Recall': result['recall'],
    'Overall_Accuracy': result['overall_accuracy'],
    'Sample_Count': result['sample_count']
} for result in performance_results.values()])

print("\nSummary of software performance metrics:")
print(performance_table.round(3))

# save
performance_table.to_csv('software_performance_comparison.csv', index=False)

# 5. heatmap
plt.figure(figsize=(10, 8))
metrics_to_plot = performance_table.drop(['Software', 'Sample_Count'], axis=1)
sns.heatmap(metrics_to_plot.T, annot=True, cmap='YlOrRd', 
            yticklabels=metrics_to_plot.columns, 
            xticklabels=performance_table['Software'])
plt.title('Heatmap of various software performance metrics')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# 7. Pay special attention to the lack of Pspire.
pspire_missing_count = df['pspire'].isnull().sum()
pspire_total_count = len(df)
print(f"\npspire missing information:")
print(f"Total number of samples: {pspire_total_count}")
print(f"number of missing samples: {pspire_missing_count}")
print(f"percentage of missing samples: {pspire_missing_count/pspire_total_count*100:.2f}%")

if 'pspire' in performance_results:
    pspire_result = performance_results['pspire']
    print(f"pspire valid sample size: {pspire_result['sample_count']}")
    print(f"pspire AUC: {pspire_result['roc_auc']:.3f}")
    print(f"pspire F1 score: {pspire_result['f1_score']:.3f}")
else:
    print("pspire no valid data")
