import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train= pd.read_csv(r'F:\PredictMyHouse\train.csv')
test=pd.read_csv(r'F:\PredictMyHouse\test.csv')
print('train info:',train.info())
print('train abstract:',train.describe())
missing=train.isnull().sum()
missing=missing[missing>0].sort_values(ascending=False)
print('missing values:',missing.head(10))
correlation=train.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
print('max correlation with H price:',correlation.head(10))
plt.figure(figsize=(10,8))
sns.heatmap(train.corr(numeric_only=True)[['SalePrice']].sort_values(by='SalePrice',ascending=False),annot=True,cmap='coolwarm')
plt.title('correlation with SalePrice')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
features=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
x=train[features]
y=train['SalePrice']
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
y_pred_lr=lr_model.predict(x_test)
mae_lr=mean_absolute_error(y_test,y_pred_lr)
r2_lr=r2_score(y_test,y_pred_lr)
print("Linear Regression:")
print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"R^2 Score: {r2_lr:.2f}")
rf_model=RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(x_train,y_train)
y_pred_rf=rf_model.predict(x_test)
mae_rf=mean_absolute_error(y_test,y_pred_rf)
r2_rf=r2_score(y_test,y_pred_rf)
print('RandomForestRegressor:')
print(f"MAE-RF: {mae_rf:.2f}")
print(f"R^2-RF: {r2_rf:.2f}")
print('model comparison:')
print(f"Linear Regression=MAE: {mae_lr:.2f}, R^2: {r2_lr:.2f}")
print(f"Random Forest=MAE: {mae_rf:.2f}, R^2: {r2_rf:.2f}")
comparison=x_test.copy()
comparison['ActualPrice-lr:']=y_test
comparison['PredictedPrice-lr:']=y_pred_lr
print(comparison.head())
comparison = x_test.copy()
comparison['ActualPrice'] = y_test
comparison['PredictedPrice_RF'] = y_pred_rf
print("\nSample Predictions (Random Forest):")
print(comparison.head())
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test,y=y_pred_rf,alpha=0.6,color='blue', label='predictions(Random Forest)')
sns.lineplot(x=y_test, y=y_test,color='red', label='ideal prediction')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Random Forest: Actual vs Predicted SalePrice')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
errors = abs(y_pred_rf - y_test)
plt.figure(figsize=(10,5))
sns.histplot(errors, bins=30, kde=True, color='purple')
plt.title("Distribution of Prediction Errors (Random Forest)")
plt.xlabel("Error (|Predicted - Actual|)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
import joblib
import os
model_path = os.path.join(os.getcwd(), "house_price_model.pkl")
joblib.dump(rf_model, model_path)
print(f"model saved in: {model_path}")
loaded_model = joblib.load(model_path)
sample_prediction = loaded_model.predict(x_test[:5])
print("sample prediction:")
print(sample_prediction)