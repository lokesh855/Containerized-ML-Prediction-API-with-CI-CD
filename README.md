# Containerized-ML-Prediction-API-with-CI-CD

A production-ready Machine Learning Image Classification API built using **FastAPI**, containerized with **Docker**, tested using **Pytest**, and automated with **GitHub Actions CI/CD**.

---

## 📌 Project Overview

This project demonstrates how to deploy a trained deep learning image classification model as a scalable REST API.

### ✅ Features

- FastAPI backend
- TensorFlow/Keras model integration
- Multi-stage Docker build
- Docker Compose setup
- Unit & Integration testing (Pytest + Mocking)
- CI/CD pipeline using GitHub Actions
- Health check endpoint
- Volume-based model updates
- Production-ready project structure

---

## 🏗️ Project Structure

```

your-ml-api/
│
├── app/
│   ├── main.py
│   └── model.py
│
├── models/
│   └── my_classifier_model.h5
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── main.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md

````

---

## ⚙️ Local Development Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-ml-api.git
cd your-ml-api
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

* **Mac/Linux**

```bash
source venv/bin/activate
```

* **Windows**

```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the API Locally

```bash
uvicorn app.main:app --reload
```

Open in browser:

```
http://localhost:8000/docs
```

---

## 🐳 Docker Setup

### Build Docker Image

```bash
docker build -t ml-image-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 ml-image-api
```

---

## 🐳 Docker Compose (Recommended)

Build and run:

```bash
docker-compose up --build
```

Stop services:

```bash
docker-compose down
```

---

## 🧪 Running Tests

Run tests locally:

```bash
pytest -v
```

Tests include:

* Health endpoint validation
* Successful prediction with mocked model
* Invalid file type handling
* Missing file upload validation

---

## 📡 API Endpoints

###  Health Check

**GET /health**

Response:

```json
{
  "status": "ok",
  "message": "API is healthy and model is loaded."
}
```

---

### Predict Image

**POST /predict**

* Accepts multipart image file (JPEG/PNG)

Response:

```json
{
  "class_label": "dog",
  "probabilities": [0.05, 0.95]
}
```

---

## 🔒 Environment Variables

Example `.env.example`:

```
MODEL_PATH=/app/models/my_classifier_model.h5
LOG_LEVEL=INFO
```

---

## 🔄 CI/CD Pipeline

This project uses **GitHub Actions** to:

* Run unit tests automatically
* Build Docker image
* Tag image with commit SHA
* Upload prediction artifacts

Workflow file:

```
.github/workflows/main.yml
```

---

## 🧠 Tech Stack

* FastAPI
* TensorFlow / Keras
* Docker
* Docker Compose
* Pytest
* GitHub Actions
* Uvicorn

