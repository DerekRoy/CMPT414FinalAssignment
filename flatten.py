import numpy as np


def is_array(potential_array):
    return isinstance(potential_array, np.ndarray) or isinstance(potential_array, list)


class flatten:
    def __init__(self, input_shape):
        self.output_shape = (np.prod(input_shape), 1)

    def out(self):
        return self.output_shape

    def flatten(self, array):
        return array.reshape(self.output_shape)
