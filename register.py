import mlflow
import mlflow.transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from mlflow.models.signature import infer_signature
import os
import boto3

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name="eu-north-1" 
)

mlflow.set_tracking_uri("http://localhost:5000")

experiment_name = "Sentiment_Analysis_Production"
try:
    # This forces the backend to bind this specific experiment name to S3
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location="s3://sentiment-analysis-storage-abhay"
    )
except Exception:
    # If it already exists, just grab it
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)

MODEL_PATH = "trained_model"
MODEL_NAME = "review-intelligence"

mlflow.set_experiment("Sentiment Analysis")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

classifier = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False
)

sample_input = ["This movie was fantastic!"]
sample_output = classifier(sample_input)

signature = infer_signature(sample_input, sample_output)

with mlflow.start_run() as run:

    print("Logging model to MLflow...")

    model_info = mlflow.transformers.log_model(
        transformers_model=classifier,
        artifact_path="model",
        signature=signature
    )

    print("Registering model...")

    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_NAME
    )

    print("\n==============================")
    print("Registration Successful!")
    print("==============================")
    print(f"Run ID      : {run.info.run_id}")
    print(f"Model Name  : {registered_model.name}")
    print(f"Version     : {registered_model.version}")