# Project structure

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
├── preprocessing/                      # All embedding methods
│   ├── fasttext.py                     # FastText embeddings
│   ├── glove.py                        # GloVe embeddings loader/wrapper
│   ├── hf_tokenizer.py                 # HuggingFace tokenizer for BERT/BERTweet
│   ├── loader.py                       # Data loading and train/valid/test splitting
│   ├── tfidf.py                        # TF-IDF vectorizer
│   └── word2vec.py                     # Word2Vec embeddings
│
├── results/                            # All produced results
│   ├── comparison_plots/               # Plots comparing multiple models/embeddings
│   ├── individual_plots/               # Single-model training/evaluation curves
│   └── submission/                     # Generated submission CSV files
│
├── saved_models/ !GITIGNORE!           # Saved models after training
│   ├── embeddings/                     # (Optional) Precomputed embeddings storage
│   ├── hf/                             # HuggingFace model checkpoints
│
├── trainers/                           # Model training implementations
│   ├── hf_trainer.py                   # HuggingFace Trainer builder
│   ├── logger_hf.py                    # Custom HF logger for loss/accuracy
│   └── sklearn_trainer.py              # Training loop for sklearn models
│
├── grid_search/                        # Hyperparameter search
│   ├── parameters_model_selection.py   # Grid search to find optimal parameters for each model and embedding
│   ├── best_model_selection_acc.csv    # Best accuracy scores for each model and for each preprocessing
│   ├── best_model_selection_f1.csv     # Best F1 scores for each model and for each preprocessing
│
├── twitter-datasets/ !GITIGNORE!       # Original dataset provided for the project
│
├── main.py                             # Main entry point to run any pipeline
├── env.yml                             # Conda environment definition
└── README.md                           # Project documentation



Vu que le modele est trop gros pour etre push sur github, ci-après le lien du drive. Sur le drive, il faut télécharger le dossier "saved_models" et le mettre dans le projet. Le chemin doit être exactement celui là pour que le code dans main.py fonctionne: saved_models/hf/vinai/bertweet-base_2250000/..., normalement en téléchargeant le dossier saved_models du drive c'est bon il n'y a qu'à le mettre dans le projet.
LIEN DU DRIVE AVEC LE MODELE FINAL: https://drive.google.com/drive/folders/1LYd9CHCcVyHs4K6S5zOLnroO1bzDtDaO?usp=sharing