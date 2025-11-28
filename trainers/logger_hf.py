from transformers import TrainerCallback

# Logger to log training and evaluation metrics during HuggingFace training
class MetricsLogger(TrainerCallback):

    def __init__(self):
        '''
        Initialize the MetricsLogger.

        Args:
            self: Instance of MetricsLogger

        Returns:
            None
        '''
        self.train_loss = []
        self.eval_loss = []
        self.eval_acc = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        '''
        Callback function called at each logging step during training.

        Args:
            self: Instance of MetricsLogger
            args: TrainingArguments object
            state: TrainerState object
            control: TrainerControl object
            logs (dict, optional): Dictionary of metrics logged at this step
                (default is None)
            **kwargs: Additional keyword arguments

        Returns:
            None
        
        '''
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
