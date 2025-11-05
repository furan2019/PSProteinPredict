import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score

# 读取数据
df = pd.read_csv("ValidationData_score_software_compare.csv")

print("CSV文件的列名:")
print(df.columns.tolist())
print("\n前几行数据:")
print(df.head())

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum())

# 提取真实标签
y_true = df.iloc[:, 1]  # 第二列，1为正样本，0为负样本

# 获取软件得分列（第3-8列）
software_columns = df.columns[2:8].tolist()
software_names = software_columns
software_scores = {name: df[name] for name in software_names}

print(f"\n真实标签列: {df.columns[1]}")
print(f"软件得分列: {software_names}")

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 创建一个函数来评估单个软件，处理缺失值
def evaluate_software(name, scores, y_true):
    """评估单个软件的性能，处理缺失值"""
    # 创建非缺失值的掩码
    non_null_mask = ~scores.isnull()
    
    if non_null_mask.sum() == 0:
        print(f"{name}: 没有有效数据")
        return None
    
    y_true_subset = y_true[non_null_mask]
    scores_subset = scores[non_null_mask]
    
    print(f"{name}: 有效样本数 = {len(scores_subset)}")
    
    # 计算ROC和AUC
    fpr, tpr, _ = roc_curve(y_true_subset, scores_subset)
    roc_auc = auc(fpr, tpr)
    
    # 计算F1分数（使用0.5作为阈值）
    threshold = 0.5
    y_pred = (scores_subset >= threshold).astype(int)
    f1 = f1_score(y_true_subset, y_pred)
    
    # 计算正样本准确性
    positive_mask = y_true_subset == 1
    if sum(positive_mask) > 0:
        pos_acc = accuracy_score(y_true_subset[positive_mask], y_pred[positive_mask])
    else:
        pos_acc = 0
    
    # 计算其他指标
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

# 1. 绘制ROC曲线并计算AUC值
plt.figure(figsize=(10, 8))
performance_results = {}

for name, scores in software_scores.items():
    result = evaluate_software(name, scores, y_true)
    if result is not None:
        performance_results[name] = result
        plt.plot(result['fpr'], result['tpr'], 
                label=f"{name} (AUC = {result['roc_auc']:.3f}, N={result['sample_count']})", 
                linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('各软件ROC曲线比较')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ROC_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 计算F1分数并绘制条形图
plt.figure(figsize=(10, 6))
f1_values = [result['f1_score'] for result in performance_results.values()]
software_names_plot = [result['name'] for result in performance_results.values()]

colors = plt.cm.Set3(np.linspace(0, 1, len(f1_values)))
bars = plt.bar(range(len(f1_values)), f1_values, color=colors)
plt.xticks(range(len(f1_values)), software_names_plot, rotation=45)
plt.ylabel('F1分数')
plt.title('各软件F1分数比较 (阈值=0.5)')
plt.ylim(0, 1)

# 在条形图上添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('F1_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 计算正样本的预测准确性并绘制条形图
plt.figure(figsize=(10, 6))
positive_acc_values = [result['positive_accuracy'] for result in performance_results.values()]

colors = plt.cm.Pastel1(np.linspace(0, 1, len(positive_acc_values)))
bars = plt.bar(range(len(positive_acc_values)), positive_acc_values, color=colors)
plt.xticks(range(len(positive_acc_values)), software_names_plot, rotation=45)
plt.ylabel('正样本准确性')
plt.title('各软件正样本预测准确性比较 (阈值=0.5)')
plt.ylim(0, 1)

# 在条形图上添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('positive_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 输出详细的性能指标表格
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

print("\n各软件性能指标汇总:")
print(performance_table.round(3))

# 保存性能指标表格
performance_table.to_csv('software_performance_comparison.csv', index=False)

# 5. 绘制综合性能热图
plt.figure(figsize=(10, 8))
metrics_to_plot = performance_table.drop(['Software', 'Sample_Count'], axis=1)
sns.heatmap(metrics_to_plot.T, annot=True, cmap='YlOrRd', 
            yticklabels=metrics_to_plot.columns, 
            xticklabels=performance_table['Software'])
plt.title('各软件性能指标热图')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 寻找最佳阈值并重新计算F1分数
print("\n寻找各软件最佳阈值:")
best_thresholds = {}
best_f1_scores = {}

for name, scores in software_scores.items():
    # 创建非缺失值的掩码
    non_null_mask = ~scores.isnull()
    if non_null_mask.sum() == 0:
        continue
        
    y_true_subset = y_true[non_null_mask]
    scores_subset = scores[non_null_mask]
    
    best_f1 = 0
    best_threshold = 0
    # 在0到1之间尝试100个阈值
    for threshold in np.linspace(0, 1, 100):
        y_pred = (scores_subset >= threshold).astype(int)
        f1 = f1_score(y_true_subset, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    best_thresholds[name] = best_threshold
    best_f1_scores[name] = best_f1
    print(f"{name}: 最佳阈值 = {best_threshold:.3f}, 最佳F1分数 = {best_f1:.3f}")

# 绘制最佳F1分数条形图
plt.figure(figsize=(10, 6))
colors = plt.cm.Set2(np.linspace(0, 1, len(best_f1_scores)))
bars = plt.bar(range(len(best_f1_scores)), list(best_f1_scores.values()), color=colors)
plt.xticks(range(len(best_f1_scores)), list(best_f1_scores.keys()), rotation=45)
plt.ylabel('最佳F1分数')
plt.title('各软件最佳F1分数比较')
plt.ylim(0, 1)

# 在条形图上添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('best_F1_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 特别关注pspire的缺失情况
pspire_missing_count = df['pspire'].isnull().sum()
pspire_total_count = len(df)
print(f"\npspire缺失情况:")
print(f"总样本数: {pspire_total_count}")
print(f"缺失样本数: {pspire_missing_count}")
print(f"缺失比例: {pspire_missing_count/pspire_total_count*100:.2f}%")

if 'pspire' in performance_results:
    pspire_result = performance_results['pspire']
    print(f"pspire有效样本数: {pspire_result['sample_count']}")
    print(f"pspire AUC: {pspire_result['roc_auc']:.3f}")
    print(f"pspire F1分数: {pspire_result['f1_score']:.3f}")
else:
    print("pspire没有有效数据")
