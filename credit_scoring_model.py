import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 1. LOAD THE DATA ---
# Ensure 'credit_risk_dataset.csv' is in the SAME folder as your script
try:
    df = pd.read_csv('credit_risk_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'credit_risk_dataset.csv' not found. Please check the file path.")
    exit()

# --- 2. DATA CLEANING ---
# Removing unrealistic outliers (Standard practice for this Kaggle dataset)
df = df[df['person_age'] <= 100]
df = df[df['person_emp_length'] <= 60]

# --- 3. DEFINE FEATURES & PIPELINE ---
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate']
categorical_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Define the Model Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# --- 4. TRAIN THE MODEL ---
X = df[numeric_features + categorical_features]
y = df['loan_status']

print("Training model... please wait.")
model_pipeline.fit(X, y)
print("Model training complete!\n")

# --- 5. INTERACTIVE USER INPUT ---
print("--- Creditworthiness Predictor ---")
try:
    user_input = {
        'person_age': float(input("Enter Age: ")),
        'person_income': float(input("Enter Annual Income: ")),
        'person_emp_length': float(input("Enter Years of Employment: ")),
        'loan_amnt': float(input("Enter Desired Loan Amount: ")),
        'loan_int_rate': float(input("Enter Expected Interest Rate: ")),
        'person_home_ownership': input("Home Ownership (RENT, OWN, MORTGAGE): ").upper(),
        'loan_intent': input("Loan Purpose (PERSONAL, EDUCATION, MEDICAL, VENTURE): ").upper(),
        'cb_person_default_on_file': input("Prior Default? (Y/N): ").upper()
    }

    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Prediction
    prediction = model_pipeline.predict(user_df)[0]
    probability = model_pipeline.predict_proba(user_df)[0]

    print("\n" + "="*30)
    if prediction == 0:
        print(f"RESULT: APPROVED")
        print(f"Probability of Repayment: {probability[0]*100:.2f}%")
    else:
        print(f"RESULT: REJECTED")
        print(f"Risk of Default: {probability[1]*100:.2f}%")
    print("="*30)

except ValueError:
    print("\nError: Please enter valid numeric values for age, income, and amounts.")