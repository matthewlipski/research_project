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

            for time_step_num in range(len(row['relationalAtt'][0])):
                time_step: TimeStep = [
                    row['relationalAtt'][0][time_step_num],
                    row['relationalAtt'][1][time_step_num],
                    row['relationalAtt'][2][time_step_num]
                ]
                time_frame.append(time_step)

                if (time_step_num + 1) % frame_size == 0:
                    time_sequence.append(time_frame)
                    time_frame = []

            class_encoding: ClassEncoding = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            class_encoding[int(float(row['classAttribute'])) - 1] = 1.0

            data_instance: DataInstance = DataInstance(time_sequence, class_encoding)
            data_set.append(data_instance)

    add_data_from_file('./datasets/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff')
    add_data_from_file('./datasets/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff')

    random.shuffle(data_set)

    return data_set
