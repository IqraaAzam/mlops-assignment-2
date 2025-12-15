"""
Training script for DVC pipeline
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data(data_path):
    """Load dataset from CSV file"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_data(df):
    """Prepare features and target"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a Random Forest model"""
    print("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy

def save_model(model, model_path):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    """Main training pipeline"""
    # File paths
    data_path = "data/dataset.csv"
    model_path = "models/model.pkl"
    
    # Load and prepare data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, model_path)
    
    print(f"Training pipeline completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()