from datetime import datetime
from typing import IO

import tensorflow as tf

from datasets import *
from models import *

if __name__ == '__main__':
    FRAME_SIZE: int = 1
    NUM_VALIDATION_FOLDS: int = 2  # TODO: Crashes when equal to 1
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 100

    data_set: DataSet = u_wave_gesture_library(FRAME_SIZE)
    data_set = normalize_data_set(data_set)
    data_set = flatten_data_set(data_set)

    folded_data_set: List[Tuple[DataSet, DataSet]] = fold_data_set(data_set, NUM_VALIDATION_FOLDS)
    data_set_info = get_data_set_info(data_set)

    training_accuracy: List[float] = []
    validation_accuracy: List[float] = []

    for i in range(len(folded_data_set)):
        model: keras.models.Model = conv_2d(data_set_info.num_features,
                                            data_set_info.num_classes,
                                            data_set_info.sequence_length,
                                            32)

        x_training = np.asarray(list(map(lambda data_instance: data_instance.time_sequence, folded_data_set[i][0])))
        y_training = np.asarray(list(map(lambda data_instance: data_instance.class_encoding, folded_data_set[i][0])))
        x_validation = np.asarray(list(map(lambda data_instance: data_instance.time_sequence, folded_data_set[i][1])))
        y_validation = np.asarray(list(map(lambda data_instance: data_instance.class_encoding, folded_data_set[i][1])))

        fold_number: int = i + 1

        print()
        print('################################################################################################')
        print('Fold', fold_number, 'validation commencing...')
        print('################################################################################################')
        print()

        history = model.fit(
            x_training,
            y_training,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        y_prediction = model(x_validation)

        fold_training_accuracy: float = history.history['categorical_accuracy'][-1]
        fold_validation_accuracy: float = 0.0

        for index in range(len(y_prediction)):
            if np.argmax(y_prediction[index]) != np.argmax(y_validation[index]):
                fold_validation_accuracy += 1

        fold_validation_accuracy = 1 - (fold_validation_accuracy / len(y_prediction))

        training_accuracy.append(fold_training_accuracy)
        validation_accuracy.append(fold_validation_accuracy)

    print()
    print('################################################################################################')
    print('Validation complete. Final training commencing...')
    print('################################################################################################')
    print()

    model: keras.models.Model = conv_2d(data_set_info.num_features,
                                        data_set_info.num_classes,
                                        data_set_info.sequence_length,
                                        32)

    x = np.asarray(list(map(lambda data_instance: data_instance.time_sequence, data_set)))
    y = np.asarray(list(map(lambda data_instance: data_instance.class_encoding, data_set)))

    history = model.fit(
        x,
        y,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    print()
    print('################################################################################################')
    print('Training complete. Saving model to file...')
    print('################################################################################################')
    print()

    # TODO: Figure out if this garbage works
    # # Save optimized model
    def representative_dataset_generator() -> List[np.ndarray]:
        # Generate values from a representative sample of the dataset
        # In this case the sample is just the whole dataset
        for time_sequence in x:
            # Each scalar value must be inside a 2D array that is wrapped in a list
            yield [np.array([time_sequence], dtype=np.float32)]


    # Convert model to TensorFlow Lite
    converter: tf.lite.TFLiteConverter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    # Default space & latency optimizations e.g. quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset_generator

    tf_lite_model = converter.convert()

    # Save model to file
    file_name: str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S') + '.tflite'
    model_file: IO = open('./models/' + file_name, 'wb')
    model_file.write(tf_lite_model)
    model_file.close()

    print()
    print('################################################################################################')
    print('Validation results:')
    print('Mean training accuracy:', round(sum(training_accuracy) / len(training_accuracy), 2))
    print('Mean validation accuracy:', round(sum(validation_accuracy) / len(validation_accuracy), 2))
    print('################################################################################################')
    print()
