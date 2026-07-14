import numpy as np
import evaluate
import mlflow

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
EXPERIMENT_NAME = "Sentiment Analysis"
REGISTERED_MODEL_NAME = "review-intelligence"

EPOCHS = 2
LEARNING_RATE = 2e-5
BATCH_SIZE = 16

print("=" * 50)
print("Loading IMDb Dataset...")
print("=" * 50)

dataset = load_dataset("imdb")

print(dataset)

print("\nLoading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizing Dataset...")

tokenized_dataset = dataset.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    ),
    batched=True,
)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

print(f"\nTraining Samples : {len(train_dataset)}")
print(f"Testing Samples  : {len(test_dataset)}")

print("\nLoading DistilBERT Model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return accuracy.compute(
        predictions=predictions,
        references=labels,
    )


training_args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="logs",
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nTrainer Created Successfully")
print("=" * 50)

print("\nStarting MLflow Experiment...")

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:

    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)

    print("\nTraining Model...\n")

    trainer.train()

    print("\nEvaluating Model...\n")

    metrics = trainer.evaluate()

    print(metrics)

    mlflow.log_metric("eval_loss", metrics["eval_loss"])
    mlflow.log_metric("eval_accuracy", metrics["eval_accuracy"])
    mlflow.log_metric("eval_runtime", metrics["eval_runtime"])
    mlflow.log_metric("eval_samples_per_second", metrics["eval_samples_per_second"])

    trainer.save_model("trained_model")
    tokenizer.save_pretrained("trained_model")

    mlflow.log_artifacts("trained_model", artifact_path="model")

    model_uri = f"runs:/{run.info.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

print("\n" + "=" * 50)
print("Training Completed Successfully!")
print("=" * 50)

print(f"Run ID        : {run.info.run_id}")
print(f"Model Name    : {registered_model.name}")
print(f"Model Version : {registered_model.version}")

print("\nModel saved in:")
print("trained_model/")