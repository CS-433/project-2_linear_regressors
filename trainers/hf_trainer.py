import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from evaluate import load
from trainers.logger_hf import MetricsLogger
from helpers.utils import ensure_dir, set_global_seed


def train_hf(model_name, tokenizer, train_ds, valid_ds, train_size, random_state=42): ##########
    """
    Train a HuggingFace transformer model
    
    Args:
        model_name (str): Name of the HuggingFace model
        tokenizer: Tokenizer object
        train_ds: Training dataset
        valid_ds: Validation dataset
        train_size (int): Size of the training dataset
    Returns:
        trainer: Trained HuggingFace Trainer object
        logger: MetricsLogger object containing training metrics
    """

    # load HF model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # to ensure determinism on MPS devices
    model.config.hidden_dropout_prob = 0.0 #####
    model.config.attention_probs_dropout_prob = 0.0 #####


    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # metrics function
    accuracy = load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

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
        bf16=False  if device.type == "mps" else None, ##### true de base mais test false pour reproducibility
        seed=random_state, ##########
        data_seed=random_state, ##########

        dataloader_num_workers=0, #####
        dataloader_pin_memory=False, #####
        dataloader_drop_last=False, #####
    )

    # logger
    logger = MetricsLogger()

    # seed
    set_global_seed(random_state)   ##########

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
