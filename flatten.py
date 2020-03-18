import numpy as np


def is_array(potential_array):
    return isinstance(potential_array, np.ndarray) or isinstance(potential_array, list)


class flatten:
    def __init__(self, input_shape):
        self.output_shape = (np.prod(input_shape),)

    def out(self):
        return self.output_shape

    def flatten(self, array, debug=False):
        if debug:
            print("Flatten:", array)

        if not is_array(array):
            if debug:
                print("Make array and return:", [array])

            return [array]

        output = []

        for element in array:
            for nested_element in self.flatten(element, debug):
                output.append(nested_element)

        if debug:
            print("flattened", array, "to", output)

        return np.array(output)
