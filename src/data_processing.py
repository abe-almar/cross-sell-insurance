# src/data_processing.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer

def get_preprocessing_pipeline(numeric_columns, categorical_columns, binary_columns):
    """Creates a ColumnTransformer for preprocessing numeric, categorical, and binary columns."""
    
    # Pipeline for numerical features
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical (non-binary) features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Pipeline for binary features
    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # ColumnTransformer to apply different preprocessing to different columns
    processing_pipeline = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_columns),
        ('cat', categorical_pipeline, categorical_columns),
        ('bin', binary_pipeline, binary_columns)
    ])

    return processing_pipeline
