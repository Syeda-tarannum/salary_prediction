# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv("Salary_Data.csv")

# Show top 10 rows
print("First 10 rows of the dataset:")
display(dataset.head(10))

# Scatter plot of Salary vs Experience
plt.figure()
plt.scatter(dataset['YearsExperience'], dataset['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Scatter Plot: Experience vs Salary')
plt.grid(True)
plt.show()
plt.savefig('Years of Experience.jpg')  # Save image

# Prepare X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=101)

# --- Linear Regression ---
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred_LR = LR.predict(X_test)

# Comparison DataFrame
diff_LR = y_test - y_pred_LR
res_df = pd.DataFrame({
    'Prediction': y_pred_LR,
    'Original Data': y_test,
    'Diff': diff_LR
})
print("Linear Regression Prediction vs Original:")
display(res_df)

# Train set plot
plt.figure()
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, LR.predict(X_train), color='red')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.grid(True)
plt.show()
plt.savefig('Years of Experience.png')

# Test set plot
plt.figure()
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, LR.predict(X_train), color='red')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# Evaluation
from sklearn import metrics
rmse_LR = np.sqrt(metrics.mean_squared_error(y_test, y_pred_LR))
r2_LR = metrics.r2_score(y_test, y_pred_LR)

print(f"Linear Regression RMSE: {rmse_LR}")
print(f"Linear Regression R² Score: {r2_LR}")

# Predict example
print(f"Predicted salary for 3 years of experience: {LR.predict([[3]])[0]}")

# --- Decision Tree Regressor ---
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
y_pred_dt = DT.predict(X_test)

# Comparison DataFrame
diff_DT = y_test - y_pred_dt
res_dt = pd.DataFrame({
    'Prediction': y_pred_dt,
    'Original Data': y_test,
    'Diff': diff_DT
})
print("Decision Tree Prediction vs Original:")
display(res_dt)

# Evaluation
rmse_DT = np.sqrt(metrics.mean_squared_error(y_test, y_pred_dt))
r2_DT = metrics.r2_score(y_test, y_pred_dt)

print(f"Decision Tree RMSE: {rmse_DT}")
print(f"Decision Tree R² Score: {r2_DT}")

# Text Representation of Decision Tree
from sklearn import tree
text_representation = tree.export_text(DT)
print("Decision Tree Structure:\n")
print(text_representation)

# Visual Tree Plot
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(DT, feature_names=['YearsExperience'], filled=True)
plt.show()
fig.savefig('DT.png')
