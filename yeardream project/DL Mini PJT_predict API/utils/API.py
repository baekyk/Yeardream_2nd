from model import model_load, flask
from preprocess import OHE_fit
from dataloader import dataload


df = dataload()


enc1= OHE_fit(df[['Airline']])
enc2= OHE_fit(df[['AirportFrom']])
enc3= OHE_fit(df[['AirportTo']])

bestmode = model_load()
flask(enc1, enc2, enc3, bestmode)
