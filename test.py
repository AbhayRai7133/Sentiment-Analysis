from transformers import AutoModelForSequenceClassification
from transformers import pipeline

model = AutoModelForSequenceClassification.from_pretrained("trained_model")
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer="trained_model"
)

result = classifier("This movie was absolutely fantastic!")
print(result)