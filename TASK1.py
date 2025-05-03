import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("tested.csv")

# Drop unused columns
df_model = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing values
# Safe and future-proof way to fill missing values
df_model["Age"] = df_model["Age"].fillna(df_model["Age"].median())
df_model["Fare"] = df_model["Fare"].fillna(df_model["Fare"].median())
df_model["Embarked"] = df_model["Embarked"].fillna(df_model["Embarked"].mode()[0])


# Encode categorical variables
label_encoders = {}
for column in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df_model[column] = le.fit_transform(df_model[column])
    label_encoders[column] = le

# Define features and target
X = df_model.drop("Survived", axis=1)
y = df_model["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
