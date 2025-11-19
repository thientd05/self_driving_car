from tensorflow import keras
from tensorflow.keras import layers
import numpy as np  

def create_model(inputShape):
    model = keras.models.Sequential([
        layers.Rescaling(1.0 / 255.0, input_shape=inputShape+(3,)),
        layers.Conv2D(24, (5,5), strides=2, padding='same', kernel_initializer='he_normal', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(36, (5,5), strides=2, padding='same', kernel_initializer='he_normal', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(48, (5,5), strides=2, padding='same', kernel_initializer='he_normal', activation='relu'),
        layers.Conv2D(64, (3,3), strides=1, padding='same', kernel_initializer='he_normal', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(78, (3,3), strides=1, padding='same', kernel_initializer='he_normal', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(200, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(20, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(2, activation='tanh', kernel_initializer='glorot_uniform')
    ])
    return model 
    