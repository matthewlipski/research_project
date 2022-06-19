import pickle
import random

from data_processing import *


def load_data_set() -> DataSet:
    """
    Creates a DataSet object from pickle files containing photodiode gesture data.

    :return: A DataSet object representing the dataset.
    """
    def load_pickle(file_path: str) -> List[List[List[int]]]:
        data = []
        with open(file_path, "rb") as open_file:
            while True:
                try:
                    data.append(pickle.load(open_file))
                except EOFError:
                    return data

    def create_data_instance(
            data: List[List[int]],
            num_gestures: int,
            gesture_index: int,
            down_sample: bool
    ) -> DataInstance:
        """
        Creates a DataInstance object from an instance of photodiode gesture data. The load_pickle function returns a
        list of such instances.

        :param data: An instance of photodiode gesture data.
        :param num_gestures: The total number of gestures that the data instance can represent.
        :param gesture_index: The index of the gesture that the data represents. The gesture that each index corresponds
        to is listed in the 'gestures' dictionary.
        :param down_sample: Whether to down-sample the data instance by 5x. This is useful for converting some 100Hz
        data found in the dataset to 20Hz, which is what the rest of the dataset is recorded at.

        :return: A DataInstance object representing the photodiode gesture data.
        """
        time_sequence: TimeSequence2D = []

        for step_num in range(len(data)):
            time_step: TimeStep = [
                float(data[step_num][0]),
                float(data[step_num][1]),
                float(data[step_num][2])
            ]

            time_sequence.append(time_step)

        class_encoding: ClassEncoding = [0.0] * num_gestures
        class_encoding[gesture_index] = 1.0

        if down_sample:
            return DataInstance(time_sequence[::5], class_encoding)

        return DataInstance(time_sequence, class_encoding)

    print()
    print('################################################################################################')
    print('Loading dataset...')
    print('################################################################################################')
    print()

    data_set: DataSet = []

    # Maps each gesture to and index.
    gestures = {
        0: 'swipe_up/',
        1: 'swipe_down/',
        2: 'swipe_left/',
        3: 'swipe_right/',
        4: 'clockwise/',
        5: 'counterclockwise/',
        6: 'tap/',
        7: 'double_tap/',
        8: 'zoom_in/',
        9: 'zoom_out/'
    }

    # Loads dataset and shuffles the order of data instances for each candidate.
    for candidate_num in range(1, 50):
        if candidate_num != 5 and candidate_num != 6:
            candidate_data: DataSet = []
            for gesture_num in range(len(gestures)):
                left_hand_gesture_data: List[List[List[int]]] = load_pickle(
                    'dataset/' +
                    gestures[gesture_num] +
                    'left_hand/' +
                    'candidate_' + str(candidate_num) + '.pickle'
                )
                right_hand_gesture_data: List[List[List[int]]] = load_pickle(
                    'dataset/' +
                    gestures[gesture_num] +
                    'right_hand/' +
                    'candidate_' + str(candidate_num) + '.pickle'
                )

                candidate_gesture_data: List[List[List[int]]] = left_hand_gesture_data + right_hand_gesture_data

                for instance_num, data_instance in enumerate(candidate_gesture_data):
                    # Non-final dataset - data from candidates 1-29 recorded at 20Hz but the rest is at 100Hz.
                    if candidate_num >= 30:
                        candidate_data.append(
                            create_data_instance(data_instance, len(gestures), gesture_num, True)
                        )
                    else:
                        candidate_data.append(
                            create_data_instance(data_instance, len(gestures), gesture_num, False)
                        )

            random.shuffle(candidate_data)
            data_set += candidate_data

    return data_set
