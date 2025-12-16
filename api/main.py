"""
FastAPI ML Inference API
Provides endpoints for model predictions and health checks
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model Inference API",
    description="RESTful API for ML model predictions using RandomForest classifier",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variable
model = None
model_info = {}

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    feature1: float = Field(..., description="First feature value")
    feature2: float = Field(..., description="Second feature value") 
    feature3: float = Field(..., description="Third feature value")
    
    class Config:
        schema_extra = {
            "example": {
                "feature1": 1.2,
                "feature2": 0.5,
                "feature3": 2.1
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Model prediction (0 or 1)")
    probability: List[float] = Field(..., description="Prediction probabilities for each class")
    confidence: float = Field(..., description="Confidence score (max probability)")
    model_version: str = Field(..., description="Model version information")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Path to model file")
    model_type: str = Field(..., description="Type of loaded model")
    api_version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Health check timestamp")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    instances: List[PredictionRequest] = Field(..., description="List of instances to predict")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of instances processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

def load_model():
    """Load the trained model from disk"""
    global model, model_info
    
    model_path = "models/model.pkl"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return False
    
    try:
        model = joblib.load(model_path)
        model_info = {
            "path": model_path,
            "type": str(type(model).__name__),
            "loaded_at": datetime.now().isoformat(),
            "file_size": os.path.getsize(model_path)
        }
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model type: {model_info['type']}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    logger.info("Starting MLOps API...")
    success = load_model()
    if not success:
        logger.warning("API started without model - predictions will not be available")
    else:
        logger.info("API started successfully with model loaded")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MLOps Model Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=model_info.get("path", "N/A"),
        model_type=model_info.get("type", "N/A"),
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check the health endpoint."
        )
    
    try:
        # Prepare input data
        input_data = np.array([[request.feature1, request.feature2, request.feature3]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0].tolist()
        confidence = max(probabilities)
        
        logger.info(f"Prediction made: {prediction} with confidence {confidence:.4f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probabilities,
            confidence=confidence,
            model_version=model_info.get("type", "unknown"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check the health endpoint."
        )
    
    start_time = datetime.now()
    
    try:
        predictions = []
        
        for instance in request.instances:
            # Prepare input data
            input_data = np.array([[instance.feature1, instance.feature2, instance.feature3]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0].tolist()
            confidence = max(probabilities)
            
            predictions.append(PredictionResponse(
                prediction=int(prediction),
                probability=probabilities,
                confidence=confidence,
                model_version=model_info.get("type", "unknown"),
                timestamp=datetime.now().isoformat()
            ))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction completed: {len(predictions)} instances in {processing_time:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info_endpoint():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get model parameters if available
        model_params = {}
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
        
        return {
            "model_info": model_info,
            "model_parameters": model_params,
            "feature_names": ["feature1", "feature2", "feature3"],
            "target_classes": [0, 1],
            "api_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Reload the model from disk"""
    success = load_model()
    if success:
        return {"message": "Model reloaded successfully", "timestamp": datetime.now().isoformat()}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation at /docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please check the logs for more details"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)