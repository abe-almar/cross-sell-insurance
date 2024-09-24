# main.py

import pandas as pd
from src.data_processing import get_preprocessing_pipeline
from src.custom_transformers import RemoveCollinearFeatures
from src.model_training import build_full_pipeline, tune_model
from src.feature_importance import extract_feature_importance

# Sample data loading 
data = pd.read_csv('clientes.csv')

# Define columns 
numeric_columns = ['Age', 'Annual_Premium']
binary_columns = ['Previously_Insured', 'Driving_License']
categorical_columns = ['Gender', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']

# Split into features and target
X = data.drop(columns=['Response', 'id']) 
y = data['Response']

# Get preprocessing pipeline
processing_pipeline = get_preprocessing_pipeline(numeric_columns, categorical_columns, binary_columns)

# Build the full pipeline with collinearity removal and PCA
full_pipeline = build_full_pipeline(processing_pipeline, RemoveCollinearFeatures(threshold=0.9), pca_n_components=0.95)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight':['balanced']
}

# Tune the model
best_model, best_params, best_score = tune_model(full_pipeline, X, y, param_grid)
print(f'Best Parameters: {best_params}')
print(f'Best AUC-ROC Score: {best_score}')

# Extract and display feature importance
extract_feature_importance(full_pipeline, X, numeric_columns, categorical_columns, binary_columns)
