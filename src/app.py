from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from io import BytesIO
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MobileNet Image Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the MobileNet model
model = None

@app.on_event("startup")
async def load_model():
    global model
    logger.info("Loading MobileNet model...")
    try:
        model = MobileNet(weights='imagenet')
        logger.info("MobileNet model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model")

# Serve the HTML file directly
@app.get("/", response_class=HTMLResponse)
async def root():
    try :
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())   
    except(FileNotFoundError ):
        logger.error(f"fichier index.html not found ") 

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image using MobileNet model
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    try:
        # Read image file
        contents = await file.read()
        img = image.load_img(BytesIO(contents), target_size=(224, 224))
        
        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Ensure model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Decode and format predictions
        results = decode_predictions(predictions, top=3)[0]
        formatted_results = [
            {
                "class_id": str(class_id),
                "class_name": class_name,
                "confidence": float(score)
            } for class_id, class_name, score in results
        ]
        
        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "predictions": formatted_results
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)