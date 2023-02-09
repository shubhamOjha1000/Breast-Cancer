import classicML_models
import classicML_models_metrices
from sklearn.linear_model import LogisticRegression



data_path = '/Users/shubhamojha/Desktop/Breast Cancer/data.csv'

diagnosis = {'M':1,'B':0}

list= ['Unnamed: 32','id','diagnosis']

drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

removal_list = ['symmetry_mean','texture_se','symmetry_se','smoothness_se']

test_size = 0.36

model_name = ['C4.5','CART','RandomForest','Gaussian NaiveBayes','SVM','KNN','LogisticRegression','AdaBoost','GradientBoosting','xgb']

models = [classicML_models.model1, classicML_models.model2, classicML_models.model3, classicML_models.model4, classicML_models.model5, classicML_models.model6, classicML_models.model7, classicML_models.model8, classicML_models.model9, classicML_models.model10]

classicalML_model_metrics = [classicML_models_metrices.acc_train, classicML_models_metrices.acc_test, classicML_models_metrices.pres_train, classicML_models_metrices.pres_test, classicML_models_metrices.rec_train, classicML_models_metrices.rec_test, classicML_models_metrices.f1_train, classicML_models_metrices.f1_test, classicML_models_metrices.train_time, classicML_models_metrices.test_time]

metric_threshold = 0.95

stacking_layer_combination =  [[0,1,2,3,4], [3,1,5,6,2],[0,3,5],[0,1,3,5],[0,1,2,3,4,5,6],[3,6,1,2,4],[3,2,1,6,5],[3,1]]

stack_model_list_final_estimator = LogisticRegression() 

DL_model_metrics = ['accuracy']

DL_loss = 'binary_crossentropy'

DL_optimizer='adam'

epochs=150

batch_size=100

