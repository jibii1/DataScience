import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the CSV file
data = pd.read_csv("50_startups_sample.csv", encoding='latin1')
print(data.head())  # Check data

# Split feature and target
X = data[["R&D Spend", "Administration", "Marketing Spend", "State"]]
y = data["Profit"]

# Convert categorical 'State' to dummy variables
X = pd.get_dummies(X, columns=["State"], drop_first=True)

# Save the columns for reference
feature_columns = X.columns

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on test set
y_pred = model.predict(x_test)
print("Predictions on test set:", y_pred)

#  Take input from user 
print("\nEnter values for prediction:")

# Get user input
rd_spend = float(input("R&D Spend: "))
admin = float(input("Administration: "))
marketing = float(input("Marketing Spend: "))
state = input("State: ")

# Create a DataFrame for the new input
user_input = pd.DataFrame([{
    "R&D Spend": rd_spend,
    "Administration": admin,
    "Marketing Spend": marketing,
    "State": state
}])

# Convert 'State' to dummies (same as training)
user_input = pd.get_dummies(user_input, columns=["State"], drop_first=True)

# Ensure all columns match the training 
for col in feature_columns:
    if col not in user_input.columns:
        user_input[col] = 0  # Add missing dummy columns with 0

# Reorder columns to match the training set
user_input = user_input[feature_columns]

# Predict profit
user_pred = model.predict(user_input)
print("\nPredicted Profit: ${:,.2f}".format(user_pred[0]))
