import Choosing_best_classicalML_models
from sklearn.ensemble import StackingClassifier
import config 

estimator_list = [] 

def estimate_creator(l): 
    estm = []
    for i in l:
        estm.append((Choosing_best_classicalML_models.selected_names[i], Choosing_best_classicalML_models.selected_models[i])) 
    return estm


for i in range(len(config.stacking_layer_combination)):
    estimator_list.append(estimate_creator(config.stacking_layer_combination[i]))

stack_model_list = [] 
for i in range(len(config.stacking_layer_combination)):
    stack_model_list.append(StackingClassifier(estimators=estimator_list[i], final_estimator = config.stack_model_list_final_estimator))

stack_model_name = []
for i in range(len(config.stacking_layer_combination)):
    stack_model_name.append(str("Meta Model " + str(i+1)))







