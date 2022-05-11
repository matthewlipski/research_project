import random

import pandas as pd
from scipy.io import arff

from data_processing import *


def u_wave_gesture_library(frame_size: int = 1) -> DataSet:
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
    'u_wave': u_wave_gesture_library
}
