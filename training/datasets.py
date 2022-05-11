import pickle
import random

import pandas as pd
from scipy.io import arff

from data_processing import *


def initial(frame_size: int = 1) -> DataSet:
    data_set: DataSet = []
    data_set_path: str = './datasets/initial/'

    class_indices = {
        0: 'swipe_up/',
        1: 'swipe_down/',
        2: 'swipe_left/',
        3: 'swipe_right/'
    }

    def load_pickle(file_path: str) -> List[List[List[int]]]:
        data = []
        with open(file_path, "rb") as open_file:
            while True:
                try:
                    data.append(pickle.load(open_file))
                except EOFError:
                    return data

    def add_data(data: List[List[List[int]]], class_index: int):
        for instance in data:
            time_sequence: TimeSequence3D = []
            time_frame: TimeFrame = []

            for step_num in range(100):
                time_step: TimeStep = [
                    float(instance[step_num][0]),
                    float(instance[step_num][1]),
                    float(instance[step_num][2])
                ]

                time_frame.append(time_step)

                if (step_num + 1) % frame_size == 0:
                    time_sequence.append(time_frame)
                    time_frame = []

            class_encoding: ClassEncoding = [0.0, 0.0, 0.0, 0.0]
            class_encoding[class_index] = 1.0

            data_instance: DataInstance = DataInstance(time_sequence, class_encoding)
            data_set.append(data_instance)

    for i in range(len(class_indices)):
        left_hand_class_data = load_pickle(data_set_path + class_indices[i] + 'left_hand/data.pickle')
        add_data(left_hand_class_data, i)

        right_hand_class_data = load_pickle(data_set_path + class_indices[i] + 'right_hand/data.pickle')
        add_data(right_hand_class_data, i)

    random.shuffle(data_set)

    return data_set


def u_wave(frame_size: int = 1) -> DataSet:
    """
    Creates a DataSet object by reading the UWave Gesture library.
    :param frame_size: The number of time steps to include in each frame.
    :return: A DataSet object representing the UWave Gesture library.
    """
    data_set: DataSet = []

    def add_data_from_file(path: str):
        arff_data = arff.loadarff(path)
        df = pd.DataFrame(arff_data[0])

        for index, row in df.iterrows():
            time_sequence: TimeSequence3D = []
            time_frame: TimeFrame = []

            for time_step_num in range(1, 316):
                time_step: TimeStep = [
                    row['att' + str(time_step_num)],
                    row['att' + str(315 + time_step_num)],
                    row['att' + str(630 + time_step_num)]
                ]
                time_frame.append(time_step)

                if time_step_num % frame_size == 0:
                    time_sequence.append(time_frame)
                    time_frame = []

            class_encoding: ClassEncoding = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            class_encoding[int(row['target']) - 1] = 1.0

            data_instance: DataInstance = DataInstance(time_sequence, class_encoding)
            data_set.append(data_instance)

    print()
    print('################################################################################################')
    print('Reading dataset...')
    print('################################################################################################')
    print()

    add_data_from_file('./datasets/u_wave/UWaveGestureLibraryAll_TRAIN.arff')
    add_data_from_file('./datasets/u_wave/UWaveGestureLibraryAll_TEST.arff')

    random.shuffle(data_set)

    return data_set


load_data_set = {
    'initial': initial,
    'u_wave': u_wave
}
