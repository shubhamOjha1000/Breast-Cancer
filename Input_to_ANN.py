import pandas as pd 
import Stacking_ClassicalML_models
import metaModel_metrices
import config

creator_train = {}
for i in range(len(config.stacking_layer_combination)):
    creator_train[Stacking_ClassicalML_models.stack_model_name[i]] = metaModel_metrices.train_prediction[i] 
    
df_ann_train = pd.DataFrame(creator_train)



creator_test = {}
for i in range(len(config.stacking_layer_combination)):
    creator_test[Stacking_ClassicalML_models.stack_model_name[i]] = metaModel_metrices.test_prediction[i] 

df_ann_test = pd.DataFrame(creator_test)






