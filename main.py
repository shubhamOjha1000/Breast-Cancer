import DLmodel_dispatcher
import config
import Input_to_ANN
from data import x_train_12, x_test_12, y_train, y_test 





classifier = DLmodel_dispatcher.models['M1']
classifier.compile(optimizer = config.DL_optimizer, loss = config.DL_loss, metrics = config.DL_model_metrics)
classifier.fit(Input_to_ANN.df_ann.values, y_train, batch_size = config.batch_size, epochs = config.epochs)