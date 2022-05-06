from utils.Modules import *
from utils.packages import *

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


param_grid = {"hidden_layer_sizes": [(50,),(100,),(200,),(300,)], 
              "activation": ["identity", "logistic", "tanh", "relu"], 
              "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0001,0.0005,0.05,0.5,1.0]}


ANN_clf = MLPRegressor(max_iter=1000,random_state=1)

ANN_grid_search = GridSearchCV(ANN_clf, param_grid, cv=5,
                                
                                  return_train_score=True,
                                  verbose=True,
                                  n_jobs=-1)

ANN_grid_search.fit(X_train, y_train)

print('Best parameters found:\n', ANN_grid_search.best_params_)
stds = ANN_grid_search.cv_results_['std_test_score']
stds
