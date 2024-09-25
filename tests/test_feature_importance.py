import pytest  # type: ignore
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.feature_importance import extract_feature_importance
from src.model_training import build_full_pipeline, tune_model
from src.data_processing import get_preprocessing_pipeline
from src.custom_transformers import RemoveCollinearFeatures


@pytest.fixture
def sample_training_data():
    """Sample training data as a pandas DataFrame for the feature importance test."""
    X_train = pd.DataFrame(
        {
            "Age": [30, 40],
            "Annual_Premium": [5000, 6000],
            "Previously_Insured": [1, 0],
            "Driving_License": [0, 1],
            "Gender": ["Male", "Female"],
            "Region_Code": ["1", "2"],
            "Vehicle_Age": ["1-2", "2-3"],
            "Vehicle_Damage": ["Yes", "No"],
            "Policy_Sales_Channel": ["123", "456"],
        }
    )
    y_train = np.array([1, 0])
    return X_train, y_train


def test_feature_importance(sample_training_data):
    """Test that the feature importance extraction returns the correct number of features

    Args:
        sample_training_data (_type_): Sample training data for the pipeline
    """
    X_train, y_train = sample_training_data
    # Define the columns for preprocessing
    numeric_columns = ["Age", "Annual_Premium"]
    binary_columns = ["Previously_Insured", "Driving_License"]
    categorical_columns = [
        "Gender",
        "Region_Code",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Policy_Sales_Channel",
    ]
    # Define the param grid
    param_grid = {
        "classifier__n_estimators": [10, 20],  # Keep it small for unit testing
        "classifier__max_depth": [2, 4],
        "class_weight": ["balanced"],
    }
    # build the pipeline

    processing_pipeline = get_preprocessing_pipeline(
        numeric_columns, categorical_columns, binary_columns
    )
    full_pipeline = build_full_pipeline(
        processing_pipeline,
        RemoveCollinearFeatures(threshold=0.9),
        pca_n_components=0.95,
    )

    # Set up StratifiedKFold with 2 splits for testing
    skf = StratifiedKFold(n_splits=2)

    # Train the model using grid search with cross-validation
    trained_model = tune_model(full_pipeline, X_train, y_train, param_grid, cv=skf)

    # extract feature importance
    all_features = numeric_columns + categorical_columns + binary_columns
    feature_importance = extract_feature_importance(
        trained_model, X_train, numeric_columns, categorical_columns, binary_columns
    )

    # ensure the feature importance extraction returns the correct number of features
    assert len(feature_importance) == len(all_features)
