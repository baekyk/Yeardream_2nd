from model import model_load, flask
from preprocess import OHE_fit
from dataloader import dataload


df = dataload()

batch_size = 2048
epoch = 30
# saved_name = f'"batch :"{batch_size}, "epoch :" {epoch}'

enc1= OHE_fit(df[['Airline']])
enc2= OHE_fit(df[['AirportFrom']])
enc3= OHE_fit(df[['AirportTo']])

bestmodel = model_load(batch_size, epoch)
flask(enc1, enc2, enc3, bestmodel)
