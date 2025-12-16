"""
Test script for the FastAPI ML inference API
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing MLOps FastAPI...")
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return
    
    # Test 2: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        health_data = response.json()
        print(f"   Status: {health_data['status']}")
        print(f"   Model loaded: {health_data['model_loaded']}")
        print(f"   Model type: {health_data['model_type']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 3: Single prediction
    try:
        prediction_data = {
            "feature1": 1.2,
            "feature2": 0.5,
            "feature3": 2.1
        }
        response = requests.post(f"{base_url}/predict", json=prediction_data)
        print(f"✅ Single prediction: {response.status_code}")
        pred_result = response.json()
        print(f"   Prediction: {pred_result['prediction']}")
        print(f"   Confidence: {pred_result['confidence']:.4f}")
        print(f"   Probabilities: {pred_result['probability']}")
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
    
    # Test 4: Batch prediction
    try:
        batch_data = {
            "instances": [
                {"feature1": 1.2, "feature2": 0.5, "feature3": 2.1},
                {"feature1": 0.8, "feature2": 1.3, "feature3": 1.7},
                {"feature1": 2.1, "feature2": 0.9, "feature3": 3.2}
            ]
        }
        response = requests.post(f"{base_url}/predict/batch", json=batch_data)
        print(f"✅ Batch prediction: {response.status_code}")
        batch_result = response.json()
        print(f"   Batch size: {batch_result['batch_size']}")
        print(f"   Processing time: {batch_result['processing_time_ms']:.2f}ms")
        for i, pred in enumerate(batch_result['predictions']):
            print(f"   Instance {i+1}: prediction={pred['prediction']}, confidence={pred['confidence']:.4f}")
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
    
    # Test 5: Model info
    try:
        response = requests.get(f"{base_url}/model/info")
        print(f"✅ Model info: {response.status_code}")
        model_info = response.json()
        print(f"   Model parameters: {len(model_info['model_parameters'])} params")
        print(f"   Feature names: {model_info['feature_names']}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    print("\n🎉 API testing completed!")
    print(f"📖 API Documentation: {base_url}/docs")
    print(f"🔍 Alternative docs: {base_url}/redoc")

if __name__ == "__main__":
    print("Please start the API server first:")
    print("uvicorn api.main:app --reload")
    print("\nThen run this test script in another terminal.")
    print("Waiting 3 seconds before testing...")
    time.sleep(3)
    test_api()