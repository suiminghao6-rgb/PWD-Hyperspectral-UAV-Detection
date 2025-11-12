import pandas as pd
from skrebate import ReliefF


# 读取数据
data = pd.read_csv("")
X = data.drop(['FID', 'gridcode'], axis=1)
y = data['gridcode']
X = X.astype(float)
y = y.astype(float)

# 使用ReliefF算法进行特征选择
relieff = ReliefF()
relieff.fit(X.values, y.values)

# 提取特征重要性
feature_importances = relieff.feature_importances_

# 保存所有特征得分
all_features_df = pd.DataFrame({
    'Feature': X.columns,
    'Score': feature_importances
})
all_features_df.to_csv("", index=False)

print("All feature scores saved to .csv")

