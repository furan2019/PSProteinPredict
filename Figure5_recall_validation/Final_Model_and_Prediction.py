import numpy as np
from lightgbm import LGBMClassifier
import pandas as pd
import os
from baggingPU import BaggingClassifierPU
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample


########################### 正训练集：植物正样本+校正正样本+预测正样本：共2100个
#校正正样本700
verified_positive_samples = pd.read_csv("verified_samples_positive.csv", index_col=0) #700个

#植物抽样700
plant_positive_samples_ = pd.read_csv("positive_samples_.csv", index_col=0) #6000个
plant_positive_samples_700 = plant_positive_samples_.sample(n=len(verified_positive_samples), random_state=2025)

positive_samples_combined_ = pd.concat([plant_positive_samples_700, verified_positive_samples]) #1400个

#预测抽样700
predicted_positive_samples_ = pd.read_csv("unlabel_2_ps_new.csv", index_col=0)
mean_predicted_positive_samples_ = predicted_positive_samples_.mean()
predicted_positive_samples_fill = predicted_positive_samples_.fillna(mean_predicted_positive_samples_)

predicted_positive_samples_700 = predicted_positive_samples_fill.sample(n=len(verified_positive_samples), random_state=2025)

positive_samples_combined = pd.concat([predicted_positive_samples_700, positive_samples_combined_]) #2100个


########################## 负训练集，校正负样本+预测负样本：共2100个
#校正负样本1400
negative_valid_ = pd.read_csv("CorrectedData_negative_2121.csv", index_col=0) #2121个
mean_negative_valid = negative_valid_.mean()
negative_valid = negative_valid_.fillna(mean_negative_valid)
verified_negative_samples = negative_valid.sample(n=1400, random_state=2024) #1400个

#预测抽样700
predicted_negative_samples_ = pd.read_csv("unlabel_2_nops_new.csv", index_col=0)
mean_predicted_negative_samples_ = predicted_negative_samples_.mean()
predicted_negative_samples_fill = predicted_negative_samples_.fillna(mean_predicted_negative_samples_)

predicted_negative_samples_700 = predicted_negative_samples_fill.sample(n=700, random_state=2025)

negative_samples_combined = pd.concat([predicted_negative_samples_700, verified_negative_samples]) #2100个

############################# 合并训练集

train_data = pd.concat([positive_samples_combined, negative_samples_combined])

############################# 正验证集：校正正样本+植物正样本+发表正样本：共217个
#校正正样本 55个
verified_positive_remaining = pd.read_csv("verified_test_positive.csv", index_col=0) #55个

#植物抽样 70个
plant_positive_valid = pd.read_csv("positive_test.csv", index_col=0) #559个
plant_positive_valid_55 = plant_positive_valid.sample(n=70, random_state=2025)

positive_valid_combined_ = pd.concat([plant_positive_valid_55, verified_positive_remaining]) 

#发表正样本 92个
published_plant_ = pd.read_csv("published_plant_92.csv", index_col=0)
mean_published_plant = published_plant_.mean()
published_plant = published_plant_.fillna(mean_published_plant)

positive_valid_combined = pd.concat([published_plant, positive_valid_combined_]) #217个

############################ 负验证集:校正负样本

negative_valid_217 = negative_valid.sample(n=217, random_state=2025) #1400个

############################# 合并验证集

valid_data = pd.concat([positive_valid_combined, negative_valid_217])

############################## 训练模型并预测
base_estimator = LGBMClassifier(objective='binary', boosting_type='dart', learning_rate=0.5,
                                            bagging_fraction=0.5,
                                            feature_fraction=0.8, min_child_samples=18, num_leaves=5, random_state=2)

model = BaggingClassifierPU(base_estimator, random_state=2)
model.fit(train_data.drop('Score', axis=1), train_data['Score'])
#保存模型
model_filename = 'pu_bagging_model.pkl'
joblib.dump(model, model_filename)
print(f"模型已保存到 {model_filename}")
#加载模型
# model = joblib.load(model_filename)

# 预测概率
valid_proba = model.predict_proba(valid_data.drop('Score', axis=1))
# 预测类别（使用0.5作为阈值）
valid_predictions = model.predict(valid_data.drop('Score', axis=1))

#保存验证集每个样本的评分
valid_results = valid_data.copy()
valid_results['Predicted_Probability'] = valid_proba[:, 1]  # 正类的概率
valid_results['Predicted_Label'] = valid_predictions

valid_results_filename = 'validation_predictions.csv'
valid_results.to_csv(valid_results_filename)
print(f"验证集预测结果已保存到 {valid_results_filename}")


# 5. 评估模型性能
# 提取真实标签
y_true = valid_data['Score']
y_pred = valid_predictions

# 计算F1-score
f1 = f1_score(y_true, y_pred)
print(f"验证集F1-score: {f1:.4f}")

# 计算正样本的accuracy
# 首先提取验证集中的正样本
positive_mask = y_true == 1
y_true_positive = y_true[positive_mask]
y_pred_positive = y_pred[positive_mask]

if len(y_true_positive) > 0:
    positive_accuracy = accuracy_score(y_true_positive, y_pred_positive)
    print(f"验证集正样本Accuracy: {positive_accuracy:.4f}")
    print(f"验证集正样本数量: {len(y_true_positive)}")
else:
    print("验证集中没有正样本")

# 6. 输出更详细的分类报告
print("\n详细分类报告:")
print(classification_report(y_true, y_pred))

# 7. 输出混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# 8. 计算并输出其他指标
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n其他指标:")
print(f"精确度 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"特异度 (Specificity): {specificity:.4f}")
print(f"整体准确率: {accuracy_score(y_true, y_pred):.4f}")

# 9. 保存评估结果到文件
evaluation_results = {
    'F1_score': f1,
    'Positive_accuracy': positive_accuracy if len(y_true_positive) > 0 else 0,
    'Overall_accuracy': accuracy_score(y_true, y_pred),
    'Precision': precision,
    'Recall': recall,
    'Specificity': specificity,
    'TP': tp,
    'FP': fp,
    'TN': tn,
    'FN': fn
}

evaluation_df = pd.DataFrame([evaluation_results])
evaluation_filename = 'model_evaluation_results.csv'
evaluation_df.to_csv(evaluation_filename, index=False)
print(f"\n评估结果已保存到 {evaluation_filename}")
