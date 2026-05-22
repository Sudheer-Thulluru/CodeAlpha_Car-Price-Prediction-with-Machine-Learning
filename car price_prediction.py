import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv('car data.csv')

# Quick feature engineering
df['Car_Age'] = 2026 - df['Year']
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Define X and y
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
preds = rf.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"MSE: {mean_squared_error(y_test, preds):.2f}")
print(f"R2:  {r2_score(y_test, preds):.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # Red dashed line for perfect fit

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()