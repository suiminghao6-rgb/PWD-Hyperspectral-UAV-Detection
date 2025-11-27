import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ---------------------- 数据读取与模型训练 ----------------------
data = pd.read_csv(r"\PLSDA_ band data.csv",
                   delimiter=',', skipinitialspace=True)
X = data.drop('type', axis=1)
y = data['type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

plsda = PLSRegression(n_components=20)
plsda.fit(X_train_imputed, y_train)

# 测试集预测
y_pred_cont = plsda.predict(X_test_imputed).flatten()

# ---------------------- 概率映射与最佳阈值确定 ----------------------
# Logistic 概率函数，将连续得分压缩到 [0,1]
def sigmoid(x, k=1.0, b=0.0):
    return 1 / (1 + np.exp(-k * (x - b)))

# 映射为概率
b = np.mean(y_pred_cont)
probs = sigmoid(y_pred_cont, k=1, b=b)

# 用 ROC 曲线确定最佳概率阈值
fpr, tpr, thresholds = roc_curve(y_test, probs)
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
print(f"最佳概率阈值: {best_threshold:.4f}")

# 按阈值分类
y_pred_class = (probs >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_class))
print('混淆矩阵：\n', confusion_matrix(y_test, y_pred_class))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

# ---------------------- 读取影像并预测 ----------------------
with rasterio.open(r"\456456.tif") as src:
    image_data = src.read().astype(np.float32)
    profile = src.profile

num_bands, height, width = image_data.shape
features = image_data.reshape(num_bands, -1).T
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

features_scaled = scaler.transform(features)
features_imputed = imputer.transform(features_scaled)

# 模型预测（连续输出）
pred_scores = plsda.predict(features_imputed).flatten()

# Logistic 概率映射
pred_probs = sigmoid(pred_scores, k=1, b=b)

# ---------------------- 概率阈值分类 ----------------------
classified_pixels = (pred_probs >= best_threshold).astype(np.uint8)
classified_image = classified_pixels.reshape((height, width))

# ---------------------- 结果可视化 ----------------------
plt.figure(figsize=(6, 4))
plt.hist(pred_probs, bins=50, color='steelblue', alpha=0.7)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best threshold = {best_threshold:.3f}')
plt.legend()
plt.title('PLS-DA Probability Distribution (Logistic)')
plt.xlabel('Probability')
plt.ylabel('Pixel Count')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(classified_image, cmap='gray')
plt.title('PLS-DA 2class (Probability Threshold)')
plt.axis('off')
plt.colorbar(label='Class')
plt.show()

# ---------------------- 保存结果 ----------------------
profile['count'] = 1
profile['dtype'] = 'uint8'
out_path = r"\PLSDA_Classification_gailv.tif"

with rasterio.open(out_path, 'w', **profile) as dst:
    dst.write(classified_image, 1)

print(f"分类结果已保存至: {out_path}")
