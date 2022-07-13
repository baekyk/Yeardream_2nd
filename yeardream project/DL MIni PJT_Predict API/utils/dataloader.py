import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dataload():
    df = pd.read_csv('Airline_delay\data\Airlines.csv')
    df = df.iloc[:,1:]
    
    return df

