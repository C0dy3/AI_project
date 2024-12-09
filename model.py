import keras


def create_model():
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Input(shape=(96, 96, 3)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),

        # Second convolutional block
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),

        # Third convolutional block
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),

        # Flatten the output of the convolutional layers
        keras.layers.Flatten(),

        # Fully connected layers with dropout to reduce overfitting
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),  # 30% dropout
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),

        # Output layer
        keras.layers.Dense(5, activation='tanh')  # Final output layer
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model