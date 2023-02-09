import config


selected_names = []
selected_models = []
selected_acc = []
for i in range(len(config.model_name)): 
    if config.classicalML_model_metrics[1] >  config.metric_threshold:
        selected_names.append(config.model_name[i])
        selected_models.append(config.models[i])





        