import os
from datetime import datetime
from typing import IO
from playsound import playsound

from tensorflow import lite as tf_lite
from keras.callbacks import EarlyStopping, ModelCheckpoint

from datasets import *
from models import *


def convert_model_to_tf_lite(keras_model: keras.models.Model) -> Union[object, bytes]:
    """
    Converts an existing Keras model to TensorFlow Lite with. The only optimization used is parameter quantization, as
    quantizing activations does not affect model file size.

    :param keras_model: The Keras model to convert.

    :return: A TensorFlow Lite model with parameters quantized from floats to ints.
    """
    converter: tf_lite.TFLiteConverter = tf_lite.TFLiteConverter.from_keras_model(keras_model)

    # Includes both TensorFlow and TensorFlow
    converter.target_spec.supported_ops = [
        tf_lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf_lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    # Default space & latency optimizations i.e. parameter quantization.
    converter.optimizations = [tf_lite.Optimize.DEFAULT]

    return converter.convert()


if __name__ == '__main__':
    MODEL_NAME: str = 'rnn'
    FRAME_LENGTH: int = 10

    BATCH_SIZE: int = 32
    NUM_VALIDATION_FOLDS: int = 1

    # Callback to stop training when validation accuracy stop increasing.
    TRAINING_PATIENCE: int = 100
    early_stop_cb = EarlyStopping(
        monitor='val_categorical_accuracy',
        mode='max',
        patience=TRAINING_PATIENCE,
    )

    data_set: DataSet = load_data_set()
    data_set = normalize_data_set(data_set)
    data_set = split_data_set(data_set, FRAME_LENGTH)
    data_set = flatten_data_set(data_set)

    folded_data_set: List[Tuple[DataSet, DataSet]] = fold_data_set(data_set, NUM_VALIDATION_FOLDS)
    data_set_info = get_data_set_info(data_set)

    training_accuracy: List[float] = []
    validation_accuracy: List[float] = []

    # Maximum number of epochs spend training during validation.
    max_epochs: int = 0

    for i in range(len(folded_data_set)):
        model: keras.models.Model = create_model(MODEL_NAME,
                                                 data_set_info.num_features,
                                                 data_set_info.num_classes,
                                                 data_set_info.sequence_length,
                                                 data_set_info.frame_length)

        # Splits training & validation data into features & classes for the current fold.
        x_training = np.asarray(list(map(lambda data_instance: data_instance.time_sequence, folded_data_set[i][0])))
        y_training = np.asarray(list(map(lambda data_instance: data_instance.class_encoding, folded_data_set[i][0])))
        x_validation = np.asarray(list(map(lambda data_instance: data_instance.time_sequence, folded_data_set[i][1])))
        y_validation = np.asarray(list(map(lambda data_instance: data_instance.class_encoding, folded_data_set[i][1])))

        print()
        print('################################################################################################')
        print('Fold', i + 1, 'training commencing...')
        print('################################################################################################')
        print()

        # Trains the model on the current fold and saves its final training classification accuracy.
        fold_training_history = model.fit(
            x_training,
            y_training,
            epochs=1,
            batch_size=BATCH_SIZE,
            validation_data=(x_validation, y_validation),
            callbacks=[early_stop_cb]
        ).history

        fold_training_accuracy: float = fold_training_history['categorical_accuracy'][-1]
        training_accuracy.append(fold_training_accuracy)

        print()
        print('################################################################################################')
        print('Fold', i + 1, 'validation commencing...')
        print('################################################################################################')
        print()

        # Converts model to TensorFlow Lite.
        tf_lite_model = convert_model_to_tf_lite(model)

        # Sets up TensorFlow Lite model for inference.
        interpreter = tf_lite.Interpreter(model_content=tf_lite_model)
        interpreter.allocate_tensors()

        model_input = interpreter.get_input_details()[0]
        model_output = interpreter.get_output_details()[0]

        # Performs validation on TensorFlow Lite model.
        fold_validation_accuracy: float = 0.0

        for data_instance_index in range(len(x_validation)):
            input_data = np.array([x_validation[data_instance_index]], dtype=np.float32)

            interpreter.set_tensor(model_input['index'], input_data)
            interpreter.invoke()

            predicted_class = np.argmax(interpreter.get_tensor(model_output['index'])[0])
            actual_class = np.argmax(y_validation[data_instance_index])

            if predicted_class == actual_class:
                fold_validation_accuracy += 1.0

        fold_validation_accuracy /= len(x_validation)
        validation_accuracy.append(fold_validation_accuracy)

        max_epochs = max(max_epochs, len(fold_training_history['categorical_accuracy']))

    print()
    print('################################################################################################')
    print('Validation complete. Final training commencing...')
    print('################################################################################################')
    print()

    model: keras.models.Model = create_model(MODEL_NAME,
                                             data_set_info.num_features,
                                             data_set_info.num_classes,
                                             data_set_info.sequence_length,
                                             data_set_info.frame_length)

    # Splits training data into features & classes for the full dataset.
    x = np.asarray(list(map(lambda data_instance: data_instance.time_sequence, data_set)))
    y = np.asarray(list(map(lambda data_instance: data_instance.class_encoding, data_set)))

    # Model file name formatting.
    file_name: str = \
        MODEL_NAME + '_' + \
        str(round(sum(validation_accuracy) / len(validation_accuracy), 2)) + '_' + \
        datetime.now().strftime('%m-%d-%Y_%H-%M-%S') + \
        '.tflite'

    # Callback to save model weights at training epoch with minimum loss.
    checkpoint_file_name: str = 'temp'
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_file_name,
        monitor='loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True
    )

    # Trains the model on the full dataset.
    history = model.fit(
        x,
        y,
        epochs=max_epochs,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_cb]
    )

    # Loads model weights with minimum loss and removes the file.
    model.load_weights(checkpoint_file_name)
    os.remove(checkpoint_file_name + '.index')
    os.remove(checkpoint_file_name + '.data-00000-of-00001')

    print()
    print('################################################################################################')
    print('Training complete. Converting model to TFLite...')
    print('################################################################################################')
    print()

    tf_lite_model = convert_model_to_tf_lite(model)

    print()
    print('################################################################################################')
    print('Conversion complete. Saving model to file...')
    print('################################################################################################')
    print()

    # Saves model to file
    model_file: IO = open('./models/' + file_name, 'wb')
    model_file.write(tf_lite_model)
    model_file.close()

    print()
    print('################################################################################################')
    print('Model Performance with frame size: ' + str(FRAME_LENGTH))
    print('Mean training accuracy:', round(sum(training_accuracy) / len(training_accuracy), 2))
    print('Mean validation accuracy:', round(sum(validation_accuracy) / len(validation_accuracy), 2))
    print('Validation accuracy per fold:', validation_accuracy)
    print('################################################################################################')
    print()

    playsound('audio.mp3')
