import numpy as np
from math import ceil


def max_pool1d_single(array, ksize, strides, debug=False):
    dprint = print
    if not debug:
        dprint = lambda *args: None

    rows, cols = array.shape

    result_rows = max(1 + ceil((rows - ksize) / strides), 1)
    result_cols = max(1 + ceil((cols - ksize) / strides), 1)

    output = np.empty((result_rows, result_cols))

    curr_x, curr_y = 0, 0

    from_x, to_x = 0, ksize
    from_y, to_y = 0, ksize

    dprint(f"result_rows={result_rows}")
    dprint(f"result_cols={result_cols}")

    while curr_y < result_rows:
        while curr_x < result_cols:
            dprint(f"{from_y}:{to_y} {from_x}:{to_x}")
            dprint("arr", array[from_y:to_y, from_x:to_x])
            max_val = array[from_y:to_y, from_x:to_x].max()
            dprint("max", max_val)

            output[curr_y][curr_x] = max_val

            curr_x += 1
            from_x += strides
            to_x += strides

        from_x, to_x = 0, ksize
        from_y += strides
        to_y += strides

        curr_y += 1
        curr_x = 0

    dprint(output)
    return output


def max_pool(feature_maps, ksize, strides):
    output = []

    for i in range(feature_maps.shape[2]):
            output.append(max_pool1d_single(feature_maps[:,:,i], ksize=ksize, strides=strides))

    return output
