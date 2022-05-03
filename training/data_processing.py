from random import randrange
from typing import List, Tuple, Union
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# Single 1D time step with a value for each feature.
TimeStep = List[float]
# N 1D time steps merged into a 2D frame.
TimeFrame = List[TimeStep]
# All 1D time steps merged into a 2D sequence.
TimeSequence2D = List[TimeStep]
# All 2D frames merged into a 3D sequence.
TimeSequence3D = List[TimeFrame]
# One-hot encoding of the classes.
ClassEncoding = List[float]


# Representation of a single instance in the dataset.
class DataInstance(NamedTuple):
    # Model input data.
    time_sequence: Union[TimeSequence3D, TimeSequence2D]
    # Expected model output data.
    class_encoding: ClassEncoding


# Entire dataset, made up of all data instances.
DataSet = List[DataInstance]


class DataSetInfo(NamedTuple):
    num_features: int
    num_classes: int
    num_instances: int
    sequence_length: int
    frame_size: Union[int, None]


def get_data_set_info(data_set: DataSet) -> DataSetInfo:
    num_features: int = len(data_set[0].time_sequence[0])
    num_classes: int = len(data_set[0].class_encoding)
    num_instances: int = len(data_set)
    sequence_length: int = len(data_set[0].time_sequence)
    frame_size: Union[int, None] = None

    if type(data_set[0].time_sequence[0][0]) == list:
        num_features = len(data_set[0].time_sequence[0][0])
        frame_size = len(data_set[0].time_sequence[0])

    return DataSetInfo(num_features, num_classes, num_instances, sequence_length, frame_size)


def normalize_data_set(data_set: DataSet) -> DataSet:
    x: np.ndarray[Union[TimeSequence3D, TimeSequence2D]] = np.array(list(
        map(lambda data_instance: data_instance.time_sequence, data_set)
    ))
    y: np.ndarray[ClassEncoding] = np.array(list(
        map(lambda data_instance: data_instance.class_encoding, data_set)
    ))

    min_value: float = np.amin(x)
    max_value: float = np.amax(x)

    x = (x - min_value) / (max_value - min_value)

    normalized_data_set: DataSet = []

    for i in range(len(data_set)):
        normalized_data_set.append(DataInstance(x[i], y[i]))

    return normalized_data_set


def flatten_data_set(data_set: DataSet) -> DataSet:
    if type(data_set[0].time_sequence) == TimeSequence2D:
        return data_set

    flattened_data_set: DataSet = []

    for data_instance in data_set:
        flattened_time_sequence: TimeSequence2D = []

        for time_frame in data_instance.time_sequence:
            # 1D representation of 2D time frame - effectively a concatenation of time steps.
            frame_vector: TimeStep = []

            for time_step in time_frame:
                frame_vector.extend(time_step)

            flattened_time_sequence.append(frame_vector)

        flattened_data_set.append(DataInstance(flattened_time_sequence, data_instance.class_encoding))

    return flattened_data_set


def fold_data_set(data_set: DataSet, num_folds: int) -> List[Tuple[DataSet, DataSet]]:
    folds: List[Tuple[DataSet, DataSet]] = []

    fold_size: int = len(data_set) // num_folds

    for fold_num in range(num_folds):
        training_set: DataSet = []
        validation_set: DataSet = []

        for i in range(0, fold_num * fold_size):
            training_set.append(data_set[i])

        for i in range(fold_num * fold_size, (fold_num + 1) * fold_size):
            validation_set.append(data_set[i])

        for i in range((fold_num + 1) * fold_size, num_folds * fold_size):
            training_set.append(data_set[i])

        folds.append((training_set, validation_set))

    return folds


def plot_example_data(data_set: DataSet) -> None:
    random_index: int = randrange(len(data_set))
    data_instance = data_set[random_index]

    x_dim = []
    y_dim = []
    z_dim = []

    for time_frame in data_instance.time_sequence:
        for time_step in time_frame:
            x_dim.append(time_step[0])
            y_dim.append(time_step[1])
            z_dim.append(time_step[2])

    class_num: str = str(np.argmax(data_instance.class_encoding))

    plt.title('Example data for class ' + class_num)
    plt.plot(x_dim, 'r')
    plt.plot(y_dim, 'g')
    plt.plot(z_dim, 'b')
    plt.show()
