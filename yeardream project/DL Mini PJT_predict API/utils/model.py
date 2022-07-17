from tensorflow.keras import datasets, utils
from tensorflow.keras import models, layers, activations, initializers, losses, optimizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from flask import Flask
from flask import render_template
from flask import request
from preprocess import OHE_transform
import pandas as pd
import numpy as np
import os.path


def DL_model():
    model = models.Sequential() # Build up the "Sequence" of layers (Linear stack of layers)

    # Dense-layer (with he-initialization)
    model.add(layers.Dense(input_dim=608, units=1024, activation=None, kernel_initializer=initializers.he_uniform())) # he-uniform initialization
    model.add(layers.BatchNormalization()) # Use this line as if needed
    model.add(layers.Activation('elu')) # elu or relu (or layers.ELU / layers.LeakyReLU)


    model.add(layers.Dense(units=1024, activation=None, kernel_initializer=initializers.he_uniform())) 
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu')) 

    model.add(layers.Dense(units=512, activation=None, kernel_initializer=initializers.he_uniform())) 
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    
    model.add(layers.Dense(units=256, activation=None, kernel_initializer=initializers.he_uniform())) 
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu')) 
    model.add(layers.Dropout(rate=0.2)) # Dropout-layer

    model.add(layers.Dense(units=128, activation=None, kernel_initializer=initializers.he_uniform())) 
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu')) 
    model.add(layers.Dropout(rate=0.2)) # Dropout-layer

    model.add(layers.Dense(units=2, activation='softmax')) # Apply softmax function on model's output

    return model


def model_compile(model):
    # "Compile" the model description (Configures the model for training)
    model.compile(optimizer=optimizers.Adam(), # Please try the Adam-optimizer
                loss=losses.categorical_crossentropy, 
                metrics=[metrics.categorical_accuracy]) # Precision / Recall / F1-Score 적용하기 @ https://j.mp/3cf3lbi


def model_fit(model, train_data, train_label, batch_size, epochs, CALLBACK):
    # "Fit" the model on training data
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, validation_split=0.3,verbose=2, callbacks=CALLBACK) 

    return history



def callback(batch_size,epoch):
    CP = ModelCheckpoint(filepath=os.path.join('../models',f'{batch_size}-{epoch}','{val_categorical_accuracy:.4f}.hdf5'),
            monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Learning Rate 줄여나가기
    LR = ReduceLROnPlateau(monitor='loss',factor=0.8,patience=3, verbose=1, min_lr=1e-8)

    CALLBACK = [CP, LR]
    return CALLBACK


def model_load(batch_size, epoch):
    MoelWeights = os.listdir(os.path.join('../models',f'{batch_size}-{epoch}'))
    BestModel = MoelWeights[ len(MoelWeights) -1 ]
    model = models.load_model(os.path.join('../models',f'{batch_size}-{epoch}',BestModel))

    return model


def flask(enc1,enc2, enc3, model):
    app = Flask(__name__)

    @app.route('/')
    @app.route('/Airline_delay')
    def AirlinePrediction():
        Airline = request.args.get("Airline")
        Flight = request.args.get("Flight")
        AirportFrom = request.args.get("AirportFrom")
        AirportTo = request.args.get("AirportTo")
        DayOfWeek = request.args.get("DayOfWeek")
        Time = request.args.get("Time")
        Length = request.args.get("Length")
        
        if Airline == None or Flight == None:
            return render_template('Airline.html', Output = '')
        
        Input = pd.DataFrame({
            'Airline': [str(Airline)],
            'Flight':  [float(Flight)] ,
            'AirportFrom': [str(AirportFrom)],
            'AirportTo': [str(AirportTo)] ,
            'DayOfWeek':  [float(DayOfWeek)] ,
            'Time':  [float(Time)] ,
            'Length':  [float(Length)]
        })
        
        Input = OHE_transform(Input[['Airline']],Input,enc1)
        Input = OHE_transform(Input[['AirportFrom']],Input,enc2)
        Input = OHE_transform(Input[['AirportTo']],Input,enc3)

        Input = Input.drop(columns=['Airline','AirportFrom','AirportTo'])
        
        # ModelOutput = model.predict(Input)[0][0]
        ModelOutput = np.argmax(model.predict(Input), axis=1)
        
        return render_template('Airline.html', Output = ModelOutput)
    
    app.run(host='0.0.0.0', port=5000)