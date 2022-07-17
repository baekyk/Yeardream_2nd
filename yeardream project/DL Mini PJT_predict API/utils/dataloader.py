import pandas as pd

def dataload():
    df = pd.read_csv('..\data\Airlines.csv')
    df = df.iloc[:,1:]
    
    return df

