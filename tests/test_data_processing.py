import pytest  # type: ignore
import numpy as np
import pandas as pd
from src.data_processing import get_preprocessing_pipeline


@pytest.fixture
def sample_data():
    """Sample input data as a pandas DataFrame with missing values, categorical, and numeric features."""
    data = {
        "Age": [30, 40],
        "Annual_Premium": [5000, np.nan],
        "Previously_Insured": [1, 0],
        "Driving_License": [0, 1],
        "Gender": ["Male", "Female"],
        "Region_Code": ["1", "2"],
        "Vehicle_Age": ["1-2", "2-3"],
        "Vehicle_Damage": ["Yes", "No"],
        "Policy_Sales_Channel": ["123", "456"],
    }
    return pd.DataFrame(data)


def test_data_processing_pipeline(sample_data):
    """Test that the data processing pipeline correctly handles missing values and categorical features.

    Args:
        sample_data (np.array): Sample imput data with missing values, categorical and numeric features.
    """
    numeric_columns = ["Age", "Annual_Premium"]
    categorical_columns = [
        "Gender",
        "Region_Code",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Policy_Sales_Channel",
    ]
    binary_columns = ["Previously_Insured", "Driving_License"]

    pipeline = get_preprocessing_pipeline(
        numeric_columns, categorical_columns, binary_columns
    )

    # transform the data
    transformed_data = pipeline.fit_transform(sample_data)

    # Test the transformed data has the expected shape
    assert transformed_data.shape[1] > len(
        numeric_columns
    )  # OneHotEncoding expands categorical features into multiple columns
    assert not np.isnan(
        transformed_data
    ).any()  # ensures no missing values after transformation
