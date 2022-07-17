from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import model_selection
from tensorflow.keras import utils
import pandas as pd
import numpy as np


def OHE_fit(data):
    OHE = OneHotEncoder(handle_unknown='ignore')
    enc = OHE.fit(data)
    return enc

# transform & DataFrame concat
def OHE_transform(column,data,enc):
    trans = enc.transform(column)
    con = pd.DataFrame(trans.toarray(), columns=enc.categories_[0])
    data = pd.concat([data,con],axis=1)

    return data


def train_test_split(df):
    test = df.loc[:,'Delay']
    train = df.drop(columns=['Delay','Airline','AirportFrom','AirportTo'])

    train_data, test_data, train_label, test_label = model_selection.train_test_split(train,test,test_size=0.3, random_state=0)

    return train_data, test_data, train_label, test_label



def scaler_transform(train_data, test_data, scaler):
    train_data = pd.DataFrame(scaler.transform(train_data), columns=train_data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data),columns=train_data.columns)

    return train_data, test_data


def categorical(train_label, test_label):
    train_label = utils.to_categorical(train_label) 
    test_label = utils.to_categorical(test_label) 

    return train_label, test_label