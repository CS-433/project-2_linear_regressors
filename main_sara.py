from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison

# Parameters for data loading
train_size=20_000
valid_size=2_000

# Parameters for the logistic regression model
alpha_log=5.0
gamma_log=0.6
max_iter_log=1200

# Parameters for the multi-layer perceptron model
alpha_mlp=1.0
gamma_mlp=0.1
#learning_rate_init=0.001
#learning_rate='constant'
#shuffle=True
max_iter_mlp=1200
hidden_layer_sizes=(256, 128)
activation='relu'
random_state=42


Pipeline(
    embedding="word2vec",
    model="mlp",
    train_size=train_size,
    valid_size=valid_size,
    alpha=alpha_mlp,
    gamma=gamma_mlp,
    #learning_rate_init=learning_rate_init,
    #learning_rate=learning_rate,
    #shuffle=shuffle,
    max_iter=max_iter_mlp,
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    random_state=random_state
).run() 

plot_comparison()

#python -m main_sara  
