import pytest  # type: ignore
import pandas as pd
import numpy as np


@pytest.fixture
def correlated_data():
    """Sample input data with high collinearity between columns."""
    return np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])


@pytest.fixture
def sample_training_data():
    """
    Fixture to generate sample training data for unit testing.
    Returns a tuple (X_train, y_train) where:
    - X_train is a pandas DataFrame containing training features
    - y_train is a numpy array containing the labels
    """
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
