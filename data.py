import pandas as pd
from sklearn.model_selection import train_test_split
import config 
from sklearn.preprocessing import StandardScaler

def elt_data():
     """Extract, load and transform our data assets."""

     # Extract
     df = pd.read_csv(config.data_path)
     assert df.shape == (569, 33) , "dataset shape should be (569, 33)"


     # Load
     y = df.diagnosis
     assert y.shape == (569, ) , "Target variable shape should be (569, )"
     y.to_csv('./labels.csv' , index=False, header = None)
     x = df.drop(config.list,axis = 1)
     assert x.shape == (569, 30) , "training feature shape should be (569, 30)"
     x.to_csv('./base_training_features.csv' , index=False, header = None)
    
     # Transform :-

     # Dropping highly correlated features :-
     x_1 = x.drop(config.drop_list1,axis=1)
     assert x_1.shape == (569, 16) , "After dropping highly correlated features from traing data, data shape should be (569, 16)"
     # Removing low importance feature:-
     x_12 = x_1.drop(config.removal_list,axis=1)
     assert x_12.shape == (569, 12) , "After dropping low importance features from traing data, data shape should be (569, 12)"

     return x_12, y





def get_data_splits(x, y, test_size=0.36):
     return train_test_split(x , y, test_size=0.36, random_state=18)


def encode(target_variable, mapping):
    target_variable.replace(mapping,inplace=True)


if __name__ == "__main__":
    x, y = elt_data()
    encode(target_variable = y, mapping = config.diagnosis)
    x_train_12, x_test_12, y_train, y_test = get_data_splits(x, y, test_size = config.test_size)
    scaler = StandardScaler()
    x_train_12 = scaler.fit_transform(x_train_12) 
    x_test_12 = scaler.transform(x_test_12)

    







     

