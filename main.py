from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import uuid
import os
import logging

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLOv8 model once
try:
    model = YOLO("best.pt")
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    raise RuntimeError("Failed to load YOLO model.")

# Create a directory to store output images
OUTPUT_DIR = "static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Bone Fracture Detection API is running"}

@app.post("/detect/")
async def detect_disease(file: UploadFile = File(...)):
    """Detect bone fractures from uploaded X-ray images."""
    try:
        # Validate file type
        if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
            raise HTTPException(status_code=400, detail="Invalid file type. Upload a PNG or JPG image.")

        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Perform detection
        results = model(image)

        if results and len(results) > 0:
            # Process and save result image
            result_img = results[0].plot()
            result_img_pil = Image.fromarray(result_img)

            output_filename = f"{OUTPUT_DIR}/output_{uuid.uuid4().hex}.png"
            result_img_pil.save(output_filename)

            return JSONResponse(content={
                "success": True,
                "message": "Fracture detected",
                "output_file": output_filename
            })

        return JSONResponse(content={"success": False, "message": "No fracture detected"})

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
