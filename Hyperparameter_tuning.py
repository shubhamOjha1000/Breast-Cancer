from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from data import x_train_12, x_test_12, y_train, y_test   ########
import data
import Hyperparamters
import numpy as np



# KNN :-
acc_rate=[]
for i in range(Hyperparamters.knn[0], Hyperparamters.knn[1]):
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(data.x_train_12, data.y_train)     #########
    pred = knn.predict(x_test_12)
    acc_rate.append(np.mean(pred==y_test)) 

# KNN best parameter 
for i in range(len(acc_rate)):
    if max(acc_rate) == acc_rate[i]:
        knn_best_params = i + 1
        break




# Decision Tree :-
acc_rate_1=[]
acc_rate_2=[]
for i in range(Hyperparamters.DT[0], Hyperparamters.DT[1]):
    c4_5 = DecisionTreeClassifier(criterion="entropy", max_depth = i) 
    cart = DecisionTreeClassifier(criterion="gini", max_depth = i)
    c4_5.fit(x_train_12, y_train) 
    cart.fit(x_train_12, y_train) 
    predl=c4_5.predict(x_test_12) 
    pred2=cart.predict(x_test_12)
    acc_rate_1. append (np. mean (predl==y_test)) 
    acc_rate_2. append (np. mean (pred2==y_test))

# Decision Tree best parameter :-
for i in range(len(acc_rate_1)):
    if max(acc_rate_1) == acc_rate_1[i]:
        dt_entropy_best_params = i + 1
        break

for i in range(len(acc_rate_2)):
    if max(acc_rate_2) == acc_rate_2[i]:
        dt_gini_best_params = i + 1
        break



# SVM :-
svc = SVC() 
grid_search = GridSearchCV(svc, Hyperparamters.SVM_parameters) 
grid_search.fit(x_train_12, y_train)
SVM_best_params = grid_search.best_params_



# GradientBoosting :-
gbc = GradientBoostingClassifier() 
grid_search_gbc = GridSearchCV(gbc, Hyperparamters.GradientBoosting_parameters, cv = Hyperparamters.cv, n_jobs = Hyperparamters.n_jobs, verbose = Hyperparamters.verbose) 
grid_search_gbc.fit(x_train_12, y_train)
GradientBoosting_best_params = grid_search_gbc.best_params_
















