import keras.layers
import keras.models


def rnn(num_features: int,
        num_classes: int,
        sequence_length: int,
        neurons_per_layer: int) -> keras.models.Model:
    model: keras.models.Model = keras.models.Sequential()

    model.add(keras.layers.SimpleRNN(neurons_per_layer, input_shape=(sequence_length, num_features)))
    # model.add(keras.layers.Conv1D(neurons_per_layer, 3, input_shape=(sequence_length, num_features)))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(neurons_per_layer))
    # model.add(keras.layers.MaxPooling1D(pool_size=2))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(num_gestures, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model
