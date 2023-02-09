# KNN :-
min_nearest_neighbor = 1
max_nearest_neighbor = 11
knn = [min_nearest_neighbor, max_nearest_neighbor]

# Decision Tree :-
min_depth = 1
max_depth = 11
DT = [min_depth, max_depth]

# SVM :-
SVM_parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1], 
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]}

# GradientBoosting :-
GradientBoosting_parameters = {
    'loss': ['deviance', 'exponential'], 
    'learning_rate': [0.001, 0.1, 1, 10], 
    'n_estimators': [100, 150, 180, 200]
}
cv = 5
n_jobs = -1
verbose = 1


# Random Forest Classifier :-
RFC_n_estimators = 60
RFC_random_state = 0

# XGB Classifier:-
random_state = 0
booster = "gbtree"












