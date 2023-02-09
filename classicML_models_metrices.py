from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.metrics import f1_score,precision_score,recall_score
from data import x_train_12, x_test_12, y_train, y_test   ########
import time
import config


acc_train = []
acc_test = []
pres_train = []
pres_test = []
rec_train = []
rec_test = []
f1_train = []
f1_test = []
train_time = []
test_time = []
confusion_matrixs = []

def classification_model_report(model,name,n):
    model = model.fit(x_train_12,y_train)
    
    start_time = time.time()
    pred_train = model.predict(x_train_12)
    end_time = time.time()
    train_time_model = end_time-start_time
    train_time.append(train_time_model)
    
    start_time = time.time()
    pred_test = model.predict(x_test_12)
    end_time = time.time()
    test_time_model = end_time-start_time
    test_time.append(test_time_model)
    
    ac_train = accuracy_score(y_train,pred_train)
    ac_test = accuracy_score(y_test,pred_test)
    acc_train.append(ac_train)
    acc_test.append(ac_test)
    
    pr_train = precision_score(y_train,pred_train)
    pr_test = precision_score(y_test, pred_test)
    pres_train.append(pr_train)
    pres_test.append(pr_test)
    
    re_train = recall_score(y_train,pred_train)
    re_test = recall_score(y_test, pred_test)
    rec_train.append(re_train)
    rec_test.append(re_test)
    
    f_train = f1_score(y_train, pred_train)
    f_test = f1_score(y_test, pred_test)
    f1_train.append(f_train)
    f1_test.append(f_test)
    
    cm = confusion_matrix(y_test,pred_test)
    confusion_matrixs.append(cm)


    # write code for each model metrics to export as csv


    #return acc_train, acc_test, pres_train, pres_test, rec_train, rec_test, f1_train, f1_test, confusion_matrixs


if __name__ == "__main__":
    for i in range(len(config.models)):
        classification_model_report(config.models[i], config.model_name[i], 0)

        # write code for each model metrics to export as csv
        # columns :- metrices,  rows :- 10 models






    

        