"""
MLOps Training Pipeline DAG
This DAG orchestrates the machine learning training pipeline with the following tasks:
1. Load data from CSV
2. Train RandomForest model
3. Save trained model
4. Log results
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'train_pipeline',
    default_args=default_args,
    description='ML Training Pipeline DAG',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    tags=['ml', 'training', 'mlops'],
)

def load_data(**context):
    """Task 1: Load data from CSV file"""
    data_path = "/opt/airflow/data/dataset.csv"
    
    logging.info(f"Loading data from {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info(f"First 5 rows:\n{df.head()}")
    
    # Save dataset info to XCom for next tasks
    dataset_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'data_path': data_path
    }
    
    return dataset_info

def prepare_and_train_model(**context):
    """Task 2: Prepare data and train the model"""
    # Get dataset info from previous task
    dataset_info = context['task_instance'].xcom_pull(task_ids='load_data')
    data_path = dataset_info['data_path']
    
    logging.info("Starting model training process...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logging.info(f"Training set size: {X_train.shape[0]}")
    logging.info(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    logging.info("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    logging.info("Model training completed")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.4f}")
    
    # Return training results
    training_results = {
        'accuracy': accuracy,
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 5
    }
    
    return training_results

def save_model(**context):
    """Task 3: Save the trained model"""
    # Get training results from previous task
    training_results = context['task_instance'].xcom_pull(task_ids='train_model')
    dataset_info = context['task_instance'].xcom_pull(task_ids='load_data')
    
    logging.info("Saving trained model...")
    
    # Recreate the model (in production, you'd pass the actual model object)
    # For this demo, we'll retrain quickly to save
    data_path = dataset_info['data_path']
    df = pd.read_csv(data_path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model
    model_path = "/opt/airflow/models/model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Verify model was saved
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        logging.info(f"Model file size: {model_size} bytes")
        
        # Test loading the model
        loaded_model = joblib.load(model_path)
        logging.info(f"Model loaded successfully. Type: {type(loaded_model)}")
        
        return {
            'model_path': model_path,
            'model_size': model_size,
            'save_status': 'success'
        }
    else:
        raise Exception("Failed to save model")

def log_results(**context):
    """Task 4: Log final results and pipeline summary"""
    # Get results from all previous tasks
    dataset_info = context['task_instance'].xcom_pull(task_ids='load_data')
    training_results = context['task_instance'].xcom_pull(task_ids='train_model')
    save_results = context['task_instance'].xcom_pull(task_ids='save_model')
    
    logging.info("=== ML PIPELINE EXECUTION SUMMARY ===")
    logging.info(f"Execution Date: {context['ds']}")
    logging.info(f"DAG Run ID: {context['dag_run'].run_id}")
    
    logging.info("--- Dataset Information ---")
    logging.info(f"Dataset Shape: {dataset_info['shape']}")
    logging.info(f"Columns: {dataset_info['columns']}")
    
    logging.info("--- Training Results ---")
    logging.info(f"Model Type: {training_results['model_type']}")
    logging.info(f"Training Accuracy: {training_results['accuracy']:.4f}")
    logging.info(f"Training Set Size: {training_results['train_size']}")
    logging.info(f"Test Set Size: {training_results['test_size']}")
    logging.info(f"Model Parameters: n_estimators={training_results['n_estimators']}, max_depth={training_results['max_depth']}")
    
    logging.info("--- Model Persistence ---")
    logging.info(f"Model Path: {save_results['model_path']}")
    logging.info(f"Model Size: {save_results['model_size']} bytes")
    logging.info(f"Save Status: {save_results['save_status']}")
    
    logging.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    
    return {
        'pipeline_status': 'completed',
        'execution_date': context['ds'],
        'final_accuracy': training_results['accuracy']
    }

# Define tasks
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=prepare_and_train_model,
    dag=dag,
)

save_model_task = PythonOperator(
    task_id='save_model',
    python_callable=save_model,
    dag=dag,
)

log_results_task = PythonOperator(
    task_id='log_results',
    python_callable=log_results,
    dag=dag,
)

# Health check task
health_check_task = BashOperator(
    task_id='health_check',
    bash_command='echo "Pipeline health check: OK" && date',
    dag=dag,
)

# Define task dependencies
health_check_task >> load_data_task >> train_model_task >> save_model_task >> log_results_task