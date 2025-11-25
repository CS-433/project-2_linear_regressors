from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison

Pipeline(
    embedding="glove",
    model="logreg",
    train_size=20_000,
    valid_size=2_000,
    alpha=5.0,
    gamma=0.6,
    max_iter=1200
).run() 

plot_comparison()

#python -m main_sara  
