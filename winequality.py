import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv('wine.csv')

X = df.drop('quality', axis=1)
y = df['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2:.2f}')


print("\nEnter wine chemical properties to predict quality:")
input_features = []
for feature in X.columns:
    val = float(input(f"{feature}: "))
    input_features.append(val)

predicted_quality = model.predict([input_features])[0]
print(f"\nPredicted Wine Quality Score: {predicted_quality:.2f}")
