from transformers import TrainerCallback

class MetricsLogger(TrainerCallback):
    """
    Logs training + evaluation metrics during HuggingFace training.
    Used for plotting learning curves after training.
    """

    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.eval_acc = []

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None:
            return

        # train loss
        if "loss" in logs:
            self.train_loss.append((state.global_step, logs["loss"]))

        # eval loss
        if "eval_loss" in logs:
            self.eval_loss.append((state.global_step, logs["eval_loss"]))

        # eval accuracy
        if "eval_accuracy" in logs:
            self.eval_acc.append((state.global_step, logs["eval_accuracy"]))
