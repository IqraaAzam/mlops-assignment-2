import pytest
import pandas as pd
import joblib
import os
import subprocess
import sys


# Fixture to ensure models directory exists for testing
@pytest.fixture(scope="session", autouse=True)
def setup_project_structure():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)


def test_data_loading():
    """Test 1: Data loading"""
    data = pd.read_csv("data/dataset.csv")
    assert not data.empty, "Dataset should not be empty"


def test_shape_validation():
    """Test 3: Shape validation Expected (10, 4) with target column."""
    data = pd.read_csv("data/dataset.csv")
    expected_shape = (10, 4)
    assert data.shape == expected_shape, (
        f"Dataset shape validation failed. Expected {expected_shape}, "
        f"got {data.shape}"
    )


def test_model_training_output():
    """Test 2: Model training (by verifying output file and content)"""
    # Run the training script (simulates successful training)
    # Assumes src/train.py is written to create models/model.pkl
    result = subprocess.run([sys.executable, 'src/train.py'],
                            capture_output=True, text=True, cwd='.')

    # Check if the script ran successfully
    assert result.returncode == 0, (
        f"Training script failed with error: {result.stderr}"
    )

    # Verify model is saved
    model_path = 'models/model.pkl'
    assert os.path.exists(model_path), f"Model file not created at {model_path}"

    # Verify model is loadable and of the correct type
    model = joblib.load(model_path)
    # The model should be a RandomForestClassifier from sklearn
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), (
        "Model should be a RandomForestClassifier."
    )