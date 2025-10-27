import joblib
import numpy as np
model = joblib.load("house_price_model.pkl")
print("welcome to house prediction")
print("please enter following details")
OverallQual = int(input("Overall Quality(1to10):"))
GrLivArea = float(input("Living Area(sq ft):"))
GarageCars = int(input("Number of Garage Cars:"))
TotalBsmtSF = float(input("Basement Area(sq ft):"))
FullBath = int(input("Number of Full Bathrooms:"))
YearBuilt = int(input("Year Built:"))

features = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]])
predicted_price = model.predict(features)[0]
print(f"Predicted Price: ${predicted_price:,.2f}")
