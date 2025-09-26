import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset5
df = pd.read_csv("score_updated.csv")
print(df.head())

# Prepare features and target variable
X = df[['Hours']]  # 2D array for sklearn
y = df['Scores']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict scores on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"Model R² Score: {r2:.2f}")

# Predict score for user input
hours = float(input("Enter number of hours studied to predict score: "))
predicted_score = model.predict([[hours]])
print(f"Predicted Score for studying {hours} hours: {predicted_score[0]:.2f}")
