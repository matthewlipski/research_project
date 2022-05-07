import keras.layers
import keras.models


def rnn(num_features: int, num_classes: int, sequence_length: int, num_neurons: int) -> keras.models.Model:
    model: keras.models.Model = keras.models.Sequential()

    model.add(keras.layers.SimpleRNN(num_neurons, input_shape=(sequence_length, num_features)))
    # model.add(keras.layers.Conv1D(neurons_per_layer, 3, input_shape=(sequence_length, num_features)))
    # model.add(keras.layers.Conv2D(neurons_per_layer, 3, input_shape=(sequence_length, num_features, 1)))
    # model.add(keras.layers.Conv2D(neurons_per_layer, 3))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(neurons_per_layer))
    # model.add(keras.layers.MaxPooling1D(pool_size=2))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(num_gestures, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def conv_1d(num_features: int, num_classes: int, sequence_length: int, num_neurons: int) -> keras.models.Model:
    model: keras.models.Model = keras.models.Sequential()

    model.add(keras.layers.Conv1D(num_neurons, 3, input_shape=(sequence_length, num_features)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def lstm(num_features: int, num_classes: int, sequence_length: int, num_neurons: int) -> keras.models.Model:
    model: keras.models.Model = keras.models.Sequential()

    model.add(keras.layers.LSTM(num_neurons, input_shape=(sequence_length, num_features)))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def gru(num_features: int, num_classes: int, sequence_length: int, num_neurons: int) -> keras.models.Model:
    model: keras.models.Model = keras.models.Sequential()

    model.add(keras.layers.GRU(num_neurons, input_shape=(sequence_length, num_features)))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def transformer(num_features: int, num_classes: int, sequence_length: int, num_neurons: int) -> keras.models.Model:
    model: keras.models.Model = keras.models.Sequential()

    def transformer_encoder(inputs):
        # Attention and Normalization
        x = keras.layers.MultiHeadAttention(
            key_dim=256, num_heads=4, dropout=0.0
        )(inputs, inputs)
        x = keras.layers.Dropout(0.0)(x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = keras.layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
        x = keras.layers.Dropout(0.0)(x)
        x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    inputs = keras.Input(shape=(sequence_length, num_features, 1))
    x = inputs
    for _ in range(1):
        x = transformer_encoder(x)

    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in [128]:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(0.0)(x)
    outputs = keras.layers.Dense(8, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def conv_2d(num_features: int, num_classes: int, sequence_length: int, num_neurons: int) -> keras.models.Model:
    model: keras.models.Model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, 2, activation='relu', input_shape=(sequence_length, num_features, 1)))
    model.add(keras.layers.Conv2D(32, 2, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 1)))
    model.add(keras.layers.Conv2D(16, kernel_size=(5, 1), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

