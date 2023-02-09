from keras.models import Sequential 
from keras.layers import Dense, Dropout
import config

base_model = Sequential(
    Dense(5, kernel_initializer='uniform', activation='relu', input_dim=len(config.stacking_layer_combination )),
    Dropout(0.1),
    Dense(3, kernel_initializer='uniform', activation='relu'),
    Dropout(0.1),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
)