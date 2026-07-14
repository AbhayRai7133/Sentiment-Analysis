from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI(
    title="Review Intelligence API",
    description="Sentiment Analysis using DistilBERT and MLflow Model Registry",
    version="1.0.0"
)

IS_RENDER = os.getenv("RENDER") is not None
loaded_model = None

if not IS_RENDER:
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        MODEL_URI = "models:/review-intelligence/1"
        print(f"Loading model from MLflow Registry: {MODEL_URI}...")
        loaded_model = mlflow.transformers.load_model(model_uri=MODEL_URI)
        print("Model loaded successfully from MLflow!")
    except Exception as e:
        print(f"Local MLflow load failed ({e}), falling back to public weights...")

# Uniformly assign the pipeline object
if loaded_model is None:
    print("Initializing production environment with public Hugging Face weights...")
    loaded_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Public pipeline loaded successfully!")

class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    label: str
    score: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": loaded_model is not None}

@app.post("/predict", response_model=ReviewResponse)
def predict_sentiment(payload: ReviewRequest):
    if not payload.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    try:
        # Directly execute the inference pipeline smoothly
        if hasattr(loaded_model, "__call__"):
            result = loaded_model(payload.review)[0]
        else:
            # Fallback wrapper handle if loaded_model acts as a container dict locally
            pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            result = pipe(payload.review)[0]
            
        label_mapping = {
            "LABEL_1": "POSITIVE",
            "LABEL_0": "NEGATIVE"
        }
        friendly_label = label_mapping.get(result["label"], result["label"])
            
        return {
            "label": friendly_label,
            "score": float(result["score"])
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))