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
import Hyperparameter_tuning
import Hyperparamters
import numpy as np



model1 = DecisionTreeClassifier(criterion="entropy", max_depth = Hyperparameter_tuning.dt_entropy_best_params)
model2 = DecisionTreeClassifier(criterion="gini", max_depth = Hyperparameter_tuning.dt_gini_best_params)
model3 = RandomForestClassifier(n_estimators = Hyperparamters.RFC_n_estimators, random_state = Hyperparamters.RFC_random_state)
model4 = GaussianNB()
model5 = SVC(Hyperparamters.SVM_best_params['C'], Hyperparamters.SVM_best_params['gamma'])
model6 = KNeighborsClassifier(Hyperparamters.knn_best_params)
model7 = LogisticRegression()
model8 = AdaBoostClassifier()
model9 = GradientBoostingClassifier(Hyperparamters.GradientBoosting_best_params['learning_rate'], Hyperparamters.GradientBoosting_best_params['loss'], Hyperparamters.GradientBoosting_best_params['n_estimators'])
model10 = xgb.XGBClassifier(random_state = Hyperparamters.random_state ,booster = Hyperparamters.booster)