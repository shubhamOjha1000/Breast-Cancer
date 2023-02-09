import time
import config
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.metrics import f1_score,precision_score,recall_score
from data import x_train_12, x_test_12, y_train, y_test 
import Stacking_ClassicalML_models

stack_acc_train = []
stack_acc_test = []
stack_pres_train = []
stack_pres_test = []
stack_rec_train = []
stack_rec_test = []
stack_f1_train = []
stack_f1_test = []
stack_train_time = []
stack_test_time = []
stack_confusion_matrixs = [] 
test_prediction = []
train_prediction = []


def stack_classification_model_report(model,name, n): 
    #((Fit the model:
    model = model.fit(x_train_12,y_train) 
    #((Make predictions on training set:
    start_time = time.time()
    pred_train = model.predict(x_train_12) 
    end_time = time.time()
    train_time_model = end_time-start_time 
    stack_train_time.append(train_time_model) 
    train_prediction.append(pred_train)
    
    start_time = time.time()
    pred_test = model.predict(x_test_12) 
    end_time = time.time()
    test_time_model = end_time-start_time 
    stack_test_time.append(test_time_model) 
    test_prediction.append(pred_test)
    
    #((Print accuracy
    
    ac_train = accuracy_score(y_train,pred_train) 
    ac_test = accuracy_score(y_test,pred_test) 
    stack_acc_train.append(ac_train)
    stack_acc_test.append(ac_test) 
    #((Print precision
    pr_train = precision_score(y_train, pred_train) 
    pr_test = precision_score(y_test, pred_test)
    
    stack_pres_train.append(pr_train) 
    stack_pres_test.append(pr_test)
    #((Print recall
    re_train = recall_score(y_train, pred_train) 
    re_test = recall_score(y_test, pred_test) 
    stack_rec_train.append(re_train)
    stack_rec_test.append(re_test) 
    #((Print fl score
    f_train = f1_score(y_train, pred_train) 
    f_test = f1_score(y_test, pred_test) 
    stack_f1_train.append(f_train) 
    stack_f1_test.append(f_test)
    
    #((confusion matrix
    cm = confusion_matrix(y_test,pred_test) 
    stack_confusion_matrixs.append(cm)




if __name__ == "__main__":
    for i in range(len(config.stacking_layer_combination)):
        stack_classification_model_report(Stacking_ClassicalML_models.stack_model_list[i], Stacking_ClassicalML_models.stack_model_name[i],0)
        # write code for each meta model metrics to export as csv
        # columns :- metrices,  rows :- 10 models


