from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison

############################### Est-ce qu'on cree un fichier séparé pour les paramètres?
# Parameters for data loading
train_size=20_000
valid_size=2_000

# Parameters for the logistic regression model
alpha_log=5.0
#gamma_log=0.0
max_iter_log=1200

# Parameters for the multi-layer perceptron model
alpha_mlp=0.1
gamma_mlp=0
#learning_rate_init=0.001
#learning_rate='constant'
#shuffle=True
max_iter_mlp=1200
hidden_layer_sizes=(256, 128)
activation='relu'
random_state=42

# Parameters for the random forest model
n_estimators=150
max_depth=20

# Parameters for the SVM model
# paramètres du SVM
C = 1.0
loss = "hinge"
max_iter_svm = 2000

# Parameters for grid search
grid_search = True

#############################################
Pipeline(
    embedding="word2vec",
    model="logreg",
    train_size=train_size,
    valid_size=valid_size,
    alpha=alpha_log,
    #learning_rate_init=learning_rate_init,
    #learning_rate=learning_rate,
    #shuffle=shuffle,
    max_iter=max_iter_log,
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    random_state=random_state,
    n_estimators=n_estimators,
    max_depth=max_depth,
    C=C,
    loss=loss,
    grid_search=grid_search,
    gamma=gamma_mlp
).run() 

plot_comparison()

#python -m main_sara  
