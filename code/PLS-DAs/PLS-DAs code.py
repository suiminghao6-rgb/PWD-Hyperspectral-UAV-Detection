
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
import rasterio
import numpy as np

# 读取数据
data = pd.read_csv(r"\PLSDA_ band data.csv", delimiter=',', skipinitialspace=True)
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

# 训练PLS-DA模型
plsda = PLSRegression(n_components=20) # 调整PLS-DA模型参数
plsda.fit(X_train_imputed, y_train)

# 测试集
y_pred = plsda.predict(X_test_imputed)
y_pred_class = np.round(np.ravel(y_pred)).astype(int)

# 测试集评估
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class))
print('训练集混淆矩阵为：\n', confusion_matrix(y_test, y_pred_class))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

# 读取原始影像
with rasterio.open(r"\temp_Original_500x500.tif") as src:
    image_data = src.read()
    profile = src.profile
num_bands, height, width = image_data.shape
features = np.zeros((height * width, num_bands), dtype=np.float32)
for band in range(num_bands):
    band_data = image_data[band, :, :]
    features[:, band] = band_data.flatten()
    plt.plot(features[:, band])

# 使用训练好的PLS-DA模型进行分类
predicted_scores = plsda.predict(features)

# 绘制分布图
plt.hist(predicted_scores, alpha=0.5)
plt.show()

# 依据阈值进行分类
threshold = 1
classified_pixels = np.where(predicted_scores > threshold, 1, 0)
classified_image = classified_pixels.reshape((height, width))

# 结果可视化
plt.imshow(classified_image, cmap='gray')
plt.colorbar(label='Class')
plt.title('Classified Image')
plt.show()

# 分类
predicted_labels = ((predicted_scores >= 1) & (predicted_scores <= 1)).astype(int) # 依据影像数据设置分类阈值
predicted_labels = np.round(predicted_scores).astype(int)
classified_image = predicted_labels.reshape((height, width))

# 保存结果
profile['count'] = 1
with rasterio.open(r"\PLSDA_Classification_Result.tif", 'w', **profile) as dst:
    dst.write(classified_image, 1)
