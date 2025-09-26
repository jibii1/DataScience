import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('StudentsPerformance.csv')

# Create pass/fail column (pass if average score >= 60)
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['pass_fail'] = (df['average_score'] >= 60).astype(int)

# Features and target
X = df[['math score', 'reading score', 'writing score']]
y = df['pass_fail']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# User input for prediction
print("\nEnter student scores to predict pass/fail:")
math = float(input("Math score: "))
reading = float(input("Reading score: "))
writing = float(input("Writing score: "))

user_data = [[math, reading, writing]]
prediction = model.predict(user_data)

if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")
