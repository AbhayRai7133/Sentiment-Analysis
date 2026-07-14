from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.transformers

app = FastAPI(
    title="Review Intelligence API",
    description="Sentiment Analysis using DistilBERT and MLflow Model Registry",
    version="1.0.0"
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/review-intelligence/1"

try:
    print(f"Loading model from MLflow Registry: {MODEL_URI}...")
    loaded_model = mlflow.transformers.load_model(model_uri=MODEL_URI)
    print("Model loaded successfully from MLflow!")
except Exception as e:
    print(f"Failed to load from MLflow Registry ({e}), falling back to public Hugging Face weights...")
    from transformers import pipeline
    # Changed from "./trained_model" to the public registry identifier
    loaded_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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
        if isinstance(loaded_model, dict) and "model" in loaded_model:
            from transformers import pipeline
            pipe = pipeline("sentiment-analysis", model=loaded_model["model"], tokenizer=loaded_model["tokenizer"])
            result = pipe(payload.review)[0]
        else:
            result = loaded_model(payload.review)[0]
            
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