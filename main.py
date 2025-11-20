from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison

Pipeline(
    embedding="vinai/bertweet-base",
    model="hf",
    train_size=20_000,
    valid_size=2_000
).run()

plot_comparison()