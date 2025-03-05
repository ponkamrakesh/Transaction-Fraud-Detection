import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("C:\\Users\\rponkam\\OneDrive - DXC Production\\Desktop\\Rakesh_DS\\DataSet\\synthetic_fraud_dataset.csv")

# Display basic dataset information
print(df.describe())
print(df.info())

# Remove unnecessary columns that do not contribute to fraud detection
df.drop(columns=['Transaction_ID', 'User_ID', 'Timestamp', 'Card_Type', 'Card_Age'], inplace=True)

# Identify numerical columns
Numerical_cols = [col for col in df.columns if df[col].dtype != 'O']
Numerical_cols.remove('Fraud_Label')  # Exclude target variable

# Display unique value counts for categorical columns
for col in df.select_dtypes(include=['O']).columns:
    print(df[col].value_counts())

# One-Hot Encoding for categorical features
Object_Cols = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Authentication_Method']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[Object_Cols])
feature_names = encoder.get_feature_names_out(Object_Cols)

# Convert encoded features into a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=feature_names)

# Merge encoded features and drop original categorical columns
df = pd.concat([df, encoded_df], axis=1)
df.drop(columns=Object_Cols, inplace=True)

# Standardize numerical features
scaler = StandardScaler()
df[Numerical_cols] = scaler.fit_transform(df[Numerical_cols])

# Split features and target variable
X = df.drop(columns=['Fraud_Label'])
y = df['Fraud_Label']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)
Y_pred = log_model.predict(X_test)

# Classification report and accuracy
print("******* Logistic Regression Classification Report *******")
print(classification_report(Y_test, Y_pred))
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=1, random_state=42)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)

# Cross-validation score
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print("Cross Validation Scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# Random Forest Classification Report
print("*** Random Forest Classifier Classification Report ***")
print(classification_report(Y_test, rf_pred))
print(f"Random Forest Accuracy: {accuracy_score(Y_test, rf_pred):.4f}")

# Overfitting Check
train_acc = accuracy_score(Y_train, log_model.predict(X_train))
test_acc = accuracy_score(Y_test, Y_pred)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

if train_acc > test_acc + 0.10:  # 10% margin to detect overfitting
    print("⚠️ Model is likely overfitting!")
else:
    print("✅ Model is generalizing well.")