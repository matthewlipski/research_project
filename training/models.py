import keras.layers
import keras.models
import tensorflow as tf


def create_model(model_type: str,
                 num_features: int,
                 num_classes: int,
                 sequence_length: int,
                 frame_length: int,
                 num_neurons: int) -> keras.models.Model:
    """
    Creates a Keras machine learning model based on the input parameters.

    :param model_type: The type of model to create. Possible options can be found in models_dict.
    :param num_features: The number of data input features.
    :param num_classes: The number of data output classes.
    :param sequence_length: The number of frames in each data instance.
    :param frame_length: The number of samples/time steps in each frame of each data instance.
    :param num_neurons: The number of neurons in the first hidden layer. Neuron counts of each subsequent hidden
        layers are also derived from this number.

    :return: A compiled and untrained Keras machine learning model.
    """
    model: keras.models.Model = keras.models.Sequential()

    def rnn():
        model.add(keras.layers.SimpleRNN(num_neurons, input_shape=(sequence_length, num_features)))

    def gru():
        model.add(keras.layers.GRU(num_neurons, input_shape=(sequence_length, num_features)))

    def lstm():
        model.add(keras.layers.LSTM(num_neurons, input_shape=(sequence_length, num_features)))
        # model.add(keras.layers.Dropout(0.5))
        # model.add(keras.layers.LSTM(num_neurons, return_sequences=True))
        # model.add(keras.layers.Dropout(0.5))
        # model.add(keras.layers.LSTM(num_neurons))

    def conv_1d():
        model.add(keras.layers.Conv1D(num_neurons, 3, input_shape=(sequence_length, num_features)))
        model.add(keras.layers.Flatten())

    def conv_2d():
        model.add(keras.layers.Conv2D(32, 2, activation='relu', input_shape=(sequence_length, num_features, 1)))
        model.add(keras.layers.Conv2D(32, 2, activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(3, 1)))
        model.add(keras.layers.Conv2D(16, kernel_size=(5, 1), activation='relu'))
        model.add(keras.layers.Flatten())

    def conv_lstm():
        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(10, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu')))
        model.add(keras.layers.Dropout(0.25))
        # model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(3, 1))))
        # model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(16, kernel_size=(5, 1), activation='relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, kernel_size=(3, 1), activation='relu')))

        # model.add(keras.layers.TimeDistributed(keras.layers.Reshape((20, 3, 1))))
        # model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 1), activation='relu')))
        # model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 1), activation='relu')))
        # model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 1))))
        # model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(16, 3, activation='relu')))
        # model.add(keras.layers.TimeDistributed(keras.layers.Reshape((20, 3, 1))))
        # model.add(keras.layers.Flatten())

        model.add(keras.layers.Reshape((20, 64)))
        model.add(keras.layers.LSTM(64))

    def transformer():
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

    models_dict = {
        'rnn': rnn,
        'gru': gru,
        'lstm': lstm,
        'conv_1d': conv_1d,
        'conv_2d': conv_2d,
        'conv_lstm': conv_lstm,
        'transformer': transformer
    }

    models_dict[model_type]()

    # Final dropout and dense layers are the same across all models.
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model
