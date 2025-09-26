import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv("score_updated.csv")
print(df.head())


X = df[['Hours']]  
y = df['Scores']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
print(f"Model R² Score: {r2:.2f}")


hours = float(input("Enter number of hours studied to predict score: "))
predicted_score = model.predict([[hours]])
print(f"Predicted Score for studying {hours} hours: {predicted_score[0]:.2f}")
