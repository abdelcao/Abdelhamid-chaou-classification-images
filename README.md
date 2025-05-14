# Abdelhamid-chaou-classification-images
# Image Classification API with MobileNet

This project is a simple web-based image classification system using a pre-trained **MobileNet** model. It provides a **FastAPI** backend for prediction and a basic **HTML** interface for uploading images.

## Project Structure

image-classifier/
├── src/
│ ├── app.py # FastAPI application
│ ├── index.html # Simple frontend to test API
├── requirements.txt # Python dependencies
├── Dockerfile # Docker image for the API
├── mobilenet-deployment.yaml # Kubernetes deployment
└── README.md # Project documentation

yaml
Copier
Modifier

## 🚀 Features

- Image classification using MobileNet (pretrained on ImageNet)
- FastAPI REST API
- Simple HTML interface (can be used independently)
- Docker container support
- Kubernetes deployment (Minikube)

## ⚙️ Getting Started

### ✅ Prerequisites

- Python 3.10+
- pip
- Docker (optional)
- Minikube & kubectl (for Kubernetes)

---

## 🔧 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/abdelcao/Abdelhamid-chaou-classification-images.git
cd Abdelhamid-chaou-classification-images

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the FastAPI app
cd src
uvicorn app:app --reload