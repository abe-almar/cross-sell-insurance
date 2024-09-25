import pytest
import numpy as np
from src.model_training import build_full_pipeline, tune_model
from src.data_processing import get_preprocessing_pipeline
from src.custom_transformers import RemoveCollinearFeatures


@pytest.fixture
def sample_training_data():
    """Sample training data for the pipeline"""
    X_train = [
        [30, 5000, 1, 0, "Male", "1", "1-2", "Yes", "123"],
        [40, np.nan, 0, 1, "Female", "2", "2-3", "No", "456"],
    ]
    y_train = np.array(1, 0)
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
    # build the pipeline

    processing_pipeline = get_preprocessing_pipeline(
        numeric_columns, categorical_columns, binary_columns
    )
    full_pipeline = build_full_pipeline(
        processing_pipeline,
        RemoveCollinearFeatures(threshold=0.9),
        pca_n_components=0.95,
    )

    # train model

    trained_model = tune_model(full_pipeline, X_train, y_train)

    # ensure pipeline is fitted
    assert trained_model.named_stepd["classifier"].is_fitted()
