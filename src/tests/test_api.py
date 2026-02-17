# src/tests/test_api.py
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
    assert response.json() == {
        "status": "ok",
        "message": "API is healthy and model is loaded."
    }


# IMPORTANT:
# Always patch where the function is USED (src.main),
# not where it is DEFINED (src.model)
@patch('src.main.predict_image')
@patch('src.main.preprocess_image')
def test_predict_success_with_mocked_model(mock_preprocess_image, mock_predict_image):
    # Configure mocks
    mock_preprocess_image.return_value = "mock_preprocessed_image_array"
    mock_predict_image.return_value = {
        "class_label": "dog",
        "probabilities": [0.05, 0.95]
    }

    # Create a dummy image file in-memory
    dummy_image = Image.new('RGB', (64, 64), color='blue')
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Send POST request
    response = client.post(
        "/predict",
        files={
            "file": ("test_image.png", img_byte_arr, "image/png")
        }
    )

    # Assertions
    assert response.status_code == 200
    assert response.json() == {
        "class_label": "dog",
        "probabilities": [0.05, 0.95]
    }

    mock_preprocess_image.assert_called_once()
    mock_predict_image.assert_called_once()


def test_predict_invalid_file_type_handling():
    response = client.post(
        "/predict",
        files={
            "file": ("document.txt", b"This is not an image.", "text/plain")
        }
    )

    assert response.status_code == 400
    assert "Only image files" in response.json()["detail"]


def test_predict_missing_file_upload():
    response = client.post(
        "/predict",
        data={}
    )

    assert response.status_code == 422
    # Make assertion case-insensitive for safety
    assert "field required" in response.json()["detail"][0]["msg"].lower()
