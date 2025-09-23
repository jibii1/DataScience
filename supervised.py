import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------- DATA LOADING & PREPROCESSING -----------------------------

# Load dataset
columns = ['buying','maint','doors','persons','lug_boot','safety','class']
df = pd.read_csv(r"C:\Users\pc37\Documents\NIV\pro\car.data", names=columns)

print("First 5 rows:\n", df.head())
print("\nClass distribution:\n", df['class'].value_counts())

# Features and target
X = df.drop(columns="class")
y = df["class"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# List of categorical feature names
categorical_features = X.columns.tolist()

# Preprocessing: one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create pipeline: preprocess + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# ----------------------------- TRAIN/TEST SPLIT & MODEL TRAINING -----------------------------

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train the model
model.fit(X_train, y_train)

# ----------------------------- EVALUATION -----------------------------

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ----------------------------- USER INPUT PREDICTION -----------------------------

print("\n--- Car Evaluation Prediction ---")

# Helper function to get validated user input
def get_user_input(prompt, options):
    while True:
        val = input(f"{prompt} {options}: ").strip().lower()
        if val in options:
            return val
        print(f"Invalid input. Please choose from {options}.")

# Define valid options
buying_options = ['vhigh', 'high', 'med', 'low']
maint_options = ['vhigh', 'high', 'med', 'low']
doors_options = ['2', '3', '4', '5more']
persons_options = ['2', '4', 'more']
lug_boot_options = ['small', 'med', 'big']
safety_options = ['low', 'med', 'high']

# Get input from user
user_input = {
    'buying': get_user_input("Buying price", buying_options),
    'maint': get_user_input("Maintenance price", maint_options),
    'doors': get_user_input("Number of doors", doors_options),
    'persons': get_user_input("Person capacity", persons_options),
    'lug_boot': get_user_input("Luggage boot size", lug_boot_options),
    'safety': get_user_input("Safety level", safety_options),
}

# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# Predict
user_pred_encoded = model.predict(user_df)[0]
user_pred_label = label_encoder.inverse_transform([user_pred_encoded])[0]

print(f"\nPredicted Car Evaluation: {user_pred_label}")
