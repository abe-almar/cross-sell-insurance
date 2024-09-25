import pytest  # type: ignore
import numpy as np
import pandas as pd
from src.model_training import build_full_pipeline, tune_model
from src.data_processing import get_preprocessing_pipeline
from src.custom_transformers import RemoveCollinearFeatures


@pytest.fixture
def sample_training_data():
    """Sample training data as a pandas DataFrame for the feature importance test."""
    X_train = pd.DataFrame(
        {
            "Age": [30, 40, 20, 40, 85],
            "Annual_Premium": [5000, 6000, 6000, 3000, 7000],
            "Previously_Insured": [1, 0, 0, 1, 0],
            "Driving_License": [0, 1, 1, 1, 1],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Region_Code": ["1", "2", "25", "25", "1"],
            "Vehicle_Age": ["1-2", "2-3", "1-2", "2-3", "2-3"],
            "Vehicle_Damage": ["Yes", "No", "Yes", "Yes", "No"],
            "Policy_Sales_Channel": ["123", "456", "123", "456", "456"],
        }
    )
    y_train = np.array([1, 0, 0, 0, 1])
    return X_train, y_train


def test_model_training(sample_training_data):
    """Test that the model pipeline trains succesfully without errors.

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

    # Train the model using grid search with cross-validation handled internally
    best_estimator, _, _ = tune_model(
        full_pipeline, X_train, y_train, param_grid, cv_splits=2
    )

    # ensure pipeline is fitted
    assert best_estimator.named_steps["classifier"].fit is not None
