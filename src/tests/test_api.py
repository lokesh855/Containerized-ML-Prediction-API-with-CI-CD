# tests/test_api.py
from fastapi.testclient import TestClient
from src.main import app
import pytest
from unittest.mock import patch
import io
from PIL import Image

client = TestClient(app)

def test_health_check_endpoint():
    # Test the /health endpoint to ensure it returns a 200 OK status
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is healthy and model is loaded."}

# Use patches to mock the heavy ML model operations during unit tests
@patch('src.model.predict_image')
@patch('src.model.preprocess_image')
def test_predict_success_with_mocked_model(mock_preprocess_image, mock_predict_image):
    # Configure mocks to return predefined values for controlled testing
    mock_preprocess_image.return_value = "mock_preprocessed_image_array" # Return a dummy preprocessed image
    mock_predict_image.return_value = {"class_label": "dog", "probabilities": [0.05, 0.95]} # Mock prediction result

    # Create a dummy image file in-memory for the test request
    dummy_image = Image.new('RGB', (64, 64), color = 'blue')
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0) # Reset stream position to the beginning

    # Send a POST request to the /predict endpoint
    response = client.post(
        "/predict", 
        files={
            "file": ("test_image.png", img_byte_arr, "image/png")
        }
    )
    # Assert the response status code and content
    assert response.status_code == 200
    assert response.json() == {"class_label": "dog", "probabilities": [0.05, 0.95]}
    # Verify that the mocked functions were called as expected
    mock_preprocess_image.assert_called_once()
    mock_predict_image.assert_called_once()

def test_predict_invalid_file_type_handling():
    # Test behavior when an unsupported file type is sent
    response = client.post(
        "/predict", 
        files={
            "file": ("document.txt", b"This is not an image.", "text/plain")
        }
    )
    # Assert that the API correctly rejects invalid file types with a 400 status
    assert response.status_code == 400
    assert "Only image files (e.g., JPEG, PNG) are allowed for prediction." in response.json()["detail"]

def test_predict_missing_file_upload():
    # Test behavior when no file is uploaded
    response = client.post(
        "/predict", 
        data={}
    )
    # FastAPI automatically handles missing required fields with a 422 Unprocessable Entity
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]
