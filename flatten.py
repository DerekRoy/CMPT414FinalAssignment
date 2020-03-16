import numpy as np


def is_array(potential_array):
    return isinstance(potential_array, np.ndarray) or isinstance(potential_array, list)


def flatten(array, debug=False):
    if debug:
        print("Flatten:", array)

    if not is_array(array):
        if debug:
            print("Make array and return:", [array])

        return [array]

    output = []

    for element in array:
        for nested_element in flatten(element, debug):
            output.append(nested_element)

    if debug:
        print("flattened", array, "to", output)

    return np.array(output)
