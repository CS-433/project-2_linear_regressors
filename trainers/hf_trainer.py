import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from evaluate import load
from trainers.logger_hf import MetricsLogger
from helpers.utils import ensure_dir


def train_hf(model_name, tokenizer, train_ds, valid_ds, train_size):

    # load HF model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # metrics function
    accuracy = load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # # directories
    # run_tag = f"{model_name.split('/')[-1]}_train{train_size}"
    # output_dir = f"results/models/{run_tag}"
    # ensure_dir("results/models")
    output_dir = f"saved_models/hf/{model_name}_{train_size}"
    ensure_dir(output_dir)

    # hf training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=200,
        report_to="none",
        no_cuda=False if device.type == "mps" else None,
        fp16=False if device.type == "mps" else None,
        bf16=True  if device.type == "mps" else None,
        seed=42,
        data_seed=42,
    )

    # logger
    logger = MetricsLogger()

    # create trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        callbacks=[logger]
    )

    # train
    trainer.train()

    # log eval
    trainer.evaluate()

    # return trainer + logger
    return trainer, logger
