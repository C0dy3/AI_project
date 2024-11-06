from keras import Sequential
from keras.src.layers import Flatten, Dense


def create_model():
    model = Sequential([
        Flatten(input_shape=(96, 96, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='tanh'),
    ])
    model.compile(loss='mse', optimizer='adam')
    return model