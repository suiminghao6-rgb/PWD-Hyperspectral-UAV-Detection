import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Diabetes dataset
diabetes = datasets.load_diabetes()


# Extract features (X) and target variable (y)
X = diabetes.data
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize PLS model with the desired number of components
n_components = 2
pls_model = PLSRegression(n_components=n_components)

# Fit the model on the training data
pls_model.fit(X_train, y_train)


# Predictions on the test set
y_pred = pls_model.predict(X_test)


# Evaluate the model performance
r_squared = pls_model.score(X_test, y_test)
print(f"R-Squared Error: {r_squared}")
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# Visualize predicted vs actual values with different colors
plt.scatter(y_test, y_pred, c='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', c='red', label='Perfect Prediction')
plt.xlabel("Actual Diabetes Progression")
plt.ylabel("Predicted Diabetes Progression")
plt.title("PLS Regression: Predicted vs Actual Diabetes Progression")
plt.legend()
plt.show()

# print(diabetes.data.shape)
# diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# diabetes_df['target'] = diabetes.target
# diabetes_df.to_excel(r'K:\WUYI\PLSDA\diabetes.xlsx')

import pandas as pd
from sklearn import datasets

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Convert the dataset to a pandas DataFrame
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Export the DataFrame to an Excel file
# diabetes_df.to_excel('diabetes_dataset.xlsx', index=False)

# Print the diabetes dataset
print(diabetes_df)


