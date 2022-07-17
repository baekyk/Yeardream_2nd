from model import DL_model, callback, model_compile, model_fit, model_load
from preprocess import OHE_transform, categorical, scaler_transform, train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess import OHE_fit
from dataloader import dataload
import tensorflow as tf

batch_size = 2048
epoch = 30
# saved_name = f'"batch :"{batch_size}, "epoch :" {epoch}'


df = dataload()

enc1 = OHE_fit(df[['Airline']])
enc2 = OHE_fit(df[['AirportFrom']])
enc3 = OHE_fit(df[['AirportTo']])

df = OHE_transform(df[['Airline']],df,enc1)
df = OHE_transform(df[['AirportFrom']],df,enc2)
df = OHE_transform(df[['AirportTo']],df,enc3)

train_data, test_data, train_label, test_label = train_test_split(df)

scaler = StandardScaler()
scaler.fit(train_data)

train_data, test_data = scaler_transform(train_data, test_data,scaler)

train_label, test_label = categorical(train_label, test_label)

model = DL_model()
model_compile(model)

CALLBACK = callback(batch_size, epoch)

with tf.device("/device:GPU:0"):
    history = model_fit(model,train_data,train_label,batch_size,epoch,CALLBACK)


bestmodel = model_load()

result = bestmodel.evaluate(test_data, test_label)

print('loss (cross-entropy) :', result[0])
print('test accuracy :', result[1])