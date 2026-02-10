import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

# 1. LOAD & CLEAN
df = pd.read_csv('credit_risk_dataset.csv')
df = df[df['person_age'] <= 90]  # Removing extreme outliers
df = df[df['person_emp_length'] <= 50]

# 2. FEATURE ENGINEERING (The Efficiency Booster)
# Adding DTI: How much of their income goes to this loan?
df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']

# 3. DEFINE FEATURES
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_to_income_ratio']
categorical_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']

X = df[numeric_features + categorical_features]
y = df['loan_status']

# 4. PREPROCESSING PIPELINE
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. THE MODEL PIPELINE (Using Gradient Boosting)
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# 6. HYPERPARAMETER TUNING (Finding the "Sweet Spot")
# We test different tree depths and learning rates
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearch runs multiple versions to find the best one
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 7. RESULTS
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nBest Parameters Found: {grid_search.best_params_}")
print("-" * 30)
print(classification_report(y_test, y_pred))
print(f"Final ROC-AUC Score: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.4f}")