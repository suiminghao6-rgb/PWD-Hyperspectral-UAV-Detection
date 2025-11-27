import pandas as pd
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import rasterio
from sklearn.metrics import confusion_matrix, recall_score, classification_report, accuracy_score
import numpy as np
from scipy.spatial.distance import mahalanobis  # ✅ 马氏距离

# ---------------------- 数据读取与模型训练 ----------------------
data = pd.read_csv(r"D:\PLSDA_ band data.csv",
                   delimiter=',', skipinitialspace=True)
X = data.drop('type', axis=1)
y = data['type']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 区分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 填补空值
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 训练 PLS-DA 模型
plsda = PLSRegression(n_components=20)
plsda.fit(X_train_imputed, y_train)

# 测试集预测与评估
y_pred = plsda.predict(X_test_imputed)
y_pred_class = np.round(np.ravel(y_pred)).astype(int)

print(classification_report(y_test, y_pred_class))
print('训练集混淆矩阵为：\n', confusion_matrix(y_test, y_pred_class))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

# ---------------------- 影像读取与分类 ----------------------
with rasterio.open(r".tif") as src:
    image_data = src.read().astype(np.float32)
    profile = src.profile

num_bands, height, width = image_data.shape
print(f"影像维度: {num_bands} 波段, {height}x{width}")

# 构建特征矩阵
features = image_data.reshape(num_bands, -1).T
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# 与训练集保持一致预处理
features_scaled = scaler.transform(features)
features_imputed = imputer.transform(features_scaled)

# ---------------------- 使用 PLS-DA 模型预测 ----------------------
# 得到 PLS 潜变量空间投影
X_train_scores = plsda.transform(X_train_imputed)
X_image_scores = plsda.transform(features_imputed)

# ---------------------- 马氏距离判别 ----------------------
# 计算各类别在潜变量空间中的均值与协方差矩阵
classes = np.unique(y_train)
mean_vectors = {}
cov_matrices = {}

for cls in classes:
    subset = X_train_scores[y_train == cls]
    mean_vectors[cls] = np.mean(subset, axis=0)
    cov_matrices[cls] = np.cov(subset, rowvar=False)

# 计算每个像元到各类别均值的马氏距离
classified_pixels = np.zeros(X_image_scores.shape[0], dtype=np.uint8)
for i, score_vec in enumerate(X_image_scores):
    distances = []
    for cls in classes:
        cov_inv = np.linalg.pinv(cov_matrices[cls])  # 逆矩阵（奇异矩阵容错）
        d = mahalanobis(score_vec, mean_vectors[cls], cov_inv)
        distances.append(d)
    classified_pixels[i] = classes[np.argmin(distances)]  # 距离最小者为所属类别

classified_image = classified_pixels.reshape((height, width))

# ---------------------- 可视化 ----------------------
plt.figure(figsize=(8, 6))
plt.imshow(classified_image, cmap='gray')
plt.title('PLS-DA 2class (Mahalanobis Distance)')
plt.colorbar(label='Class')
plt.axis('off')
plt.show()

# ---------------------- 保存结果 ----------------------
profile['count'] = 1
profile['dtype'] = 'uint8'
output_path = r"\PLSDA_Classification_Mahalanobis.tif"

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(classified_image, 1)

print(f"分类结果已保存至: {output_path}")
