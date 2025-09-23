import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

x=np.array([100,200,300,1000]).reshape(-1,1)
y=np.array([100000,200000,300000,1500000])

model=LinearRegression()
model.fit(x,y)

print("Intercept(bace price):",model.intercept_)
print("Solpe (price per sqft):", model.coef_[0])

size = 1100
predicted_price = model.predict([[size]])

print(f"Predicted price for {size} sqft = ${predicted_price[0]:2f}")