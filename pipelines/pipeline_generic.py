from preprocessing import hf_tokenizer, tfidf, fasttext, word2vec, glove
from preprocessing.loader import load_raw_data
from models import logreg, svm, random_forest, mlp, hf_classifier
#from trainers.sklearn_trainer import train_sklearn
from trainers.hf_trainer import train_hf
from helpers.plots import plot_confusion_matrix, plot_training_curves, save_metrics, plot_comparison
from helpers.utils import save_submit, ensure_dir, set_global_seed
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Pipeline:

    def __init__(self, embedding, model, train_size=2_500_000, valid_size=250_000,      alpha=None, gamma=None, max_iter=None, # logreg + mlp model ###sara###
    hidden_layer_sizes=None, activation=None, random_state=None # mlp model params ###sara###
    ):  ###sara### ajout de alpha, gamma, max_iter, hidden_layer_sizes, activation, random_state
        self.embedding = embedding
        self.model_name = model
        self.train_size = train_size
        self.valid_size = valid_size
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.random_state = random_state

    def load_data(self):
        (self.train_txt,
         self.valid_txt,
         self.test_txt,
         self.y_train,
         self.y_valid) = load_raw_data(self.train_size, self.valid_size)
        self.train_labels = self.y_train
        self.valid_labels = self.y_valid

    def preprocess(self):
        print(f"[PREPROCESS] Using embedding: {self.embedding}")

        if self.embedding == "tfidf":
            self.X_train, self.X_valid, self.X_test = tfidf.make(
                self.train_txt, self.valid_txt, self.test_txt
            )

        elif self.embedding == "fasttext":
            self.X_train, self.X_valid, self.X_test = fasttext.make(
                self.train_txt, self.valid_txt, self.test_txt
            )

        elif self.embedding == "word2vec":
            self.X_train, self.X_valid, self.X_test = word2vec.make(
                self.train_txt, self.valid_txt, self.test_txt
            )

        elif self.embedding == "glove":
            self.X_train, self.X_valid, self.X_test = glove.make(
                self.train_txt, self.valid_txt, self.test_txt
            )

        elif "bert" in self.embedding.lower():  # bertweet / roberta...
            (self.tokenizer,
            self.train_ds,
            self.valid_ds,
            self.test_ds) = hf_tokenizer.make(
                train_texts=self.train_txt,
                train_labels=self.train_labels,
                valid_texts=self.valid_txt,
                valid_labels=self.valid_labels,
                test_texts=self.test_txt,
                model_name=self.embedding
            )

            
        else:
            raise ValueError(f"Unknown embedding: {self.embedding}")

    def train(self):
        print(f"[TRAIN] Training model: {self.model_name}")

        # HuggingFace models
        if "bert" in self.embedding.lower():
            (self.trainer,
             self.logger) = train_hf(
                model_name=self.embedding,
                tokenizer=self.tokenizer,
                train_ds=self.train_ds,
                valid_ds=self.valid_ds,
                train_size=self.train_size
            )
            return

        # Sklearn models
        if self.model_name == "logreg":
            self.model = logreg.make(
                alpha=self.alpha,
                gamma=self.gamma, 
                max_iter=self.max_iter
            )
            self.model.train(self.X_train, self.y_train) ###sara###

        elif self.model_name == "svm":
            self.model = svm.make()

        elif self.model_name == "forest":
            self.model = random_forest.make()

        if self.model_name == "mlp":
            self.model = mlp.make(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self.model.fit(self.X_train, self.y_train) ###sara###


        else:
            raise ValueError(f"Unknown learning model: {self.model_name}")

        
#train_sklearn(self.model, self.X_train, self.y_train)

    def evaluate(self):
        print("[EVAL] Evaluating model...")

        # HuggingFace evaluation
        if "bert" in self.embedding.lower():

            preds = self.trainer.predict(self.valid_ds).predictions.argmax(1)

            acc = accuracy_score(self.y_valid, preds)
            prec = precision_score(self.y_valid, preds, zero_division=0)
            rec = recall_score(self.y_valid, preds, zero_division=0)
            f1 = f1_score(self.y_valid, preds, zero_division=0)

            plot_confusion_matrix(
                preds, self.y_valid,
                f"{self.embedding}",
                self.train_size
            )
            plot_training_curves(
                self.logger, self.embedding, self.train_size
            )

            save_metrics(
                name=f"{self.embedding}_{self.train_size}",
                acc=acc,
                prec=prec,
                rec=rec,
                f1=f1
            )

            print(f"[METRICS] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

            return preds

        # Sklearn evaluation
        preds = self.model.predict(self.X_valid)

        acc = accuracy_score(self.y_valid, preds)
        prec = precision_score(self.y_valid, preds, zero_division=0)
        rec = recall_score(self.y_valid, preds, zero_division=0)
        f1 = f1_score(self.y_valid, preds, zero_division=0)

        plot_confusion_matrix(
            preds,
            self.y_valid,
            f"{self.embedding}_{self.model_name}",
            self.train_size
        )

        save_metrics(
            name=f"{self.embedding}_{self.model_name}_{self.train_size}",
            acc=acc,
            prec=prec,
            rec=rec,
            f1=f1
        )

        print(f"[METRICS] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
        return preds
    
    def save(self):
        print("[SAVE] Saving predictions + model...")

        # HuggingFace save
        if "bert" in self.embedding.lower():
            test_preds = self.trainer.predict(self.test_ds).predictions.argmax(1)

            save_submit(test_preds, f"{self.embedding}_{self.train_size}")

            save_dir = f"saved_models/hf/{self.embedding}_{self.train_size}"
            #from helpers.utils import ensure_dir ###sara###
            ensure_dir(save_dir)
            self.trainer.save_model(f"saved_models/hf/{self.embedding}_{self.train_size}")
            self.tokenizer.save_pretrained(f"saved_models/hf/{self.embedding}_{self.train_size}")

            return

        # Sklearn save
        test_preds = self.model.predict(self.X_test)
        final_preds = np.array([1 if p == 1 else -1 for p in test_preds])

        save_submit(final_preds, f"{self.embedding}_{self.model_name}")

        save_dir = f"saved_models/sklearn/{self.embedding}_{self.model_name}"
        ensure_dir(save_dir)

        # model saving 
        import pickle
        path = f"saved_models/sklearn/{self.embedding}_{self.model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def run(self):
        print(f"\n========= RUNNING PIPELINE =========")
        print(f"Embedding = {self.embedding}")
        print(f"Model     = {self.model_name}")
        print(f"Train size = {self.train_size}")
        print("====================================\n")

        set_global_seed(42)
        self.load_data()
        self.preprocess()
        self.train()
        self.evaluate()
        self.save()

        print("\n=========== DONE ===========\n")
