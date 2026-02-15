from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import random

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Temporary fake logic (replace with real model later)
    is_ai = random.random() > 0.5
    confidence = random.randint(80, 99)

    ai_prob = confidence if is_ai else 100 - confidence
    real_prob = 100 - ai_prob

    return {
        "label": "AI Generated" if is_ai else "Real Image",
        "confidence": confidence,
        "ai_prob": ai_prob,
        "real_prob": real_prob
    }
