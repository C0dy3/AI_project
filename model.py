import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten



def create_model():
    model = Sequential([
        Flatten(input_shape=(96, 96, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='tanh'),
    ])
    model.compile(loss='mse', optimizer='adam')
    return model