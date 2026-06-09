from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .inference import Predictor
import os

app = FastAPI(title="CottonGuard AI Model API")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
weights_path = os.getenv("MODEL_PATH", None)
predictor = Predictor(model_path=weights_path)
# Initialize predictor
# Look for weights in the current directory or backend/weights/

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    contents = await file.read()
    result = predictor.predict(contents)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

from fastapi.staticfiles import StaticFiles
import os

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend_built")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {"message": "CottonGuard AI API is running (Frontend not found)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
