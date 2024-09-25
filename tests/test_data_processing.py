import pytest
import numpy as np
from src.data_processing import get_preprocessing_pipeline


@pytest.fixture
def sample_data():
    """Sample imput data with missing values, categorical and numeric features."""
    return {
        "numeric_columns": ["Age", "Annual_Premium"],
        "binary_columns": ["Previously_Insured", "Driving_License"],
        "categorical_columns": [
            "Gender",
            "Region_Code",
            "Vehicle_Age",
            "Vehicle_Damage",
            "Policy_Sales_Channel",
        ],
        "X": np.array(
            [
                [30, 5000, 1, 0, "Male", "1", "1-2", "Yes", "123"],
                [40, np.nan, 0, 1, "Female", "2", "2-3", "No", "456"],
            ]
        ),
    }


def test_data_processing_pipeline(sample_data):
    """Test that the data processing pipeline correctly handles missing values and categorical features.

    Args:
        sample_data (np.array): Sample imput data with missing values, categorical and numeric features.
    """
    pipeline = get_preprocessing_pipeline(
        sample_data["numeric_columns"],
        sample_data["categorical_columns"],
        sample_data["binary_columns"],
    )

    # transform the data
    transformed_data = pipeline.fit_transform(sample_data["X"])

    # Test the transformed data has the expected shape
    assert transformed_data.shape[1] > len(
        sample_data["numeric_columns"]
    )  # OneHotEncoding expands categorical features into multiple columns
    assert not np.isnan(
        transformed_data
    ).any()  # ensures no missing values after transformation
