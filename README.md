# CS-433 Machine Learning - Project 2
**Team:** `linear_regressors`

**Team members:**  
- Loïc GUENZI
- Rémy JAILLAT
- Sara MAISONHAUTE


## Getting started
### Project description
This project was carried out as part of the CS-433 Machine Learning course at EPFL.
The goal of this project is to predict whether a tweet is more likely to contain a positive or negative smiley, using only the remaining text. 
The target variable is binary, indicating a *positive (1)* or *negative (-1)* smiley in the tweet.
The dataset is composed of 2~500~000 tweets.

We explore both classical vectorizer (TF-IDF, Word2Vec, FastText, GloVe) and pre-trained transformer models (BERT, BERTweet, RoBERTa…).


The pipeline allows you to:
- Load Twitter data.
- Preprocess text using the chosen vectorizer.
- Train a machine learning model.
- Evaluate its performance.
- Save models and predictions.

Further details and analysis can be found in the accompanying report:
`project2_report.pdf`

## Project structure

```text
project/
│
├── glove_scripts/                      # Scripts to build GloVe embeddings (vocab, cooccurrence, templates…)
│   ├── build_vocab.sh
│   ├── cooc.py
│   ├── cut_vocab.sh
│   ├── glove_solution.py
│   ├── glove_template.py
│   ├── pickle_vocab.py
│   └── results/ !GITIGNORE!            # embeddings results of glove pipeline
│
├── helpers/                            # Utility functions for metrics, plots, file handling…
│   ├── metrics.py                      # Accuracy / precision / recall / F1 helpers
│   ├── plots.py                        # Confusion matrices, training curves, comparison plots
│   └── utils.py                        # Directory creation, submission saving, generic helpers
│
├── models/                             # All ML classifiers
│   ├── hf_classifier.py                # (Optional) HF-based classifier wrapper
│   ├── logreg.py                       # Logistic Regression
│   ├── mlp.py                          # Multi-layer Perceptron
│   ├── random_forest.py                # Random Forest classifier
│   └── svm.py                          # Support Vector Machine
│
├── pipelines/
│   └── pipeline_generic.py             # Main pipeline orchestrating preprocessing → training → evaluation → saving
│
├── preprocessing/                      # All vectorizing methods
│   ├── fasttext.py                     # FastText embeddings
│   ├── glove.py                        # GloVe embeddings loader/wrapper
│   ├── hf_tokenizer.py                 # HuggingFace tokenizer for BERT/BERTweet
│   ├── loader.py                       # Data loading and train/valid/test splitting
│   ├── tfidf.py                        # TF-IDF vectorizer
│   └── word2vec.py                     # Word2Vec embeddings
│
├── results/                            # All produced results
│   ├── comparison_plots/               # Plots comparing multiple models/vectorizer
│   ├── individual_plots/               # Single-model training/evaluation curves
│   └── submission/                     # Generated submission CSV files
│
├── saved_models/ !GITIGNORE!           # Saved models after training
│   ├── embeddings/                     # (Optional) Precomputed embeddings storage
│   ├── hf/                             # HuggingFace models
│   └── sklearn/                        # Sklearn models
│
├── trainers/                           # Model training implementations
│   ├── hf_trainer.py                   # HuggingFace Trainer builder
│   ├── logger_hf.py                    # Custom HF logger for loss/accuracy
│   └── sklearn_trainer.py              # Training loop for sklearn models
│
├── grid_search/                        # Hyperparameter search
│   ├── parameters_model_selection.py   # Grid search to find optimal parameters for each model and vectorizer
│   ├── parameters_model_selection.csv  # Results of the grid search after testing all models with all hyperparameters 
│   ├── best_model_selection_acc.csv    # Best accuracy scores for each model and for each preprocessing
│   ├── best_model_selection_f1.csv     # Best F1 scores for each model and for each preprocessing
│
├── twitter-datasets/ !GITIGNORE!       # Original dataset provided for the project
│
├── main.py                             # Main entry point to run any pipeline
├── env.yml                             # Conda environment definition
└── README.md                           # Project documentation
```

## Running the project
1. Place the raw data (`test_data.txt`, `train_neg_full.txt`, `train_neg.txt`, `train_pos_full.txt` and `train_pos.txt` files) in the `twitter-datasets` folder. You can download the data here https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files.

2. Create and activate the environment
```bash
# create the environment (only once) 
conda env create -f env.yml

# activate the environment
conda activate project2-ml

# verify that the right version of python is installed (should be 3.10)
python --version
which python
```

3. To run the pipeline, you can run the following command in the terminal:
```bash
python -m main
```
In the main file, you can change the *vectorizer and training model*, the *size of the training and validation set*, as well as the *hyperparameters* here:
```python
pipe = Pipeline(
    vectorizer="vinai/bertweet-base",     # "tfidf", "fasttext", "word2vec", "glove", "vinai/bertweet-base", "roberta-base", "cardiffnlp/twitter-roberta-base", "vinai/bertweet-large"
    model="hf",            # "logreg", "svm", "mlp", "random_forest", "hf"
    train_size=2250000      # to run on a smaller fraction: 2000
    valid_size=225000       # to run on a smaller fraction: 200
    spe_hyperparameters     # alpha, gamma, max_iter, hidden_layer_sizes, activation
)

pipe.run()
```

4. To get our best model (BERTweet Transformers) you have to run the main.py file but you have to comment the code above (alreday commented in this code version) because this model's training took around 12 hours, and it will save the same submit as our best submission to results/submission/submit_Final_Submission_bertweet_base_full.csv. To do so, you have to download the following HF model because it was to large to get pushed on github. The path has to be: saved_models/hf/vinai/bertweet-base_2250000/. Drive's link: https://drive.google.com/drive/folders/1LYd9CHCcVyHs4K6S5zOLnroO1bzDtDaO?usp=sharing.

5. Results are saved in:
results/ for plots and submission files
saved_models/ for trained models 

The best results we had were score 0.912 on AICrowd but after some modifications in the code in order to be 100% reproducible, the best we obtained is 0.900 (submission 304074 on AICrowd).