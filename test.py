import unittest
import tensorflow as tf
import numpy as np
from max_pool import max_pool, max_pool1d_single
from flatten import flatten
from fc import fully_connected_layer

from collections import namedtuple


class TestMaxPool(unittest.TestCase):

    def test_flatten(self):
        TestCase = namedtuple('TestCase', 'input expected_output')

        test_cases = [
            TestCase(
                input=[1, 2, 3],
                expected_output=[1, 2, 3],
            ),
            TestCase(
                input=[[[[[1]]]], 2],
                expected_output=[1, 2],
            ),
            TestCase(
                input=[[1], [2], [3]],
                expected_output=[1, 2, 3],
            ),
            TestCase(
                input=[[1, 2, 3], [4], [], [5, [6, [7, 8]]]],
                expected_output=[1, 2, 3, 4, 5, 6, 7, 8],
            ),
        ]

        for test_case in test_cases:
            is_correct = False
            try:
                flat = flatten(np.array(test_case.input).shape)

                result = flat.flatten(np.array(test_case.input))
                is_correct = np.array_equal(result, np.array(test_case.expected_output))
            finally:
                if not is_correct:
                    flatten(np.array(test_case.input).shape).flatten(np.array(test_case.input), debug=True)

            self.assertTrue(is_correct)

    def test_max_pool1d_single(self):
        TestCase = namedtuple('TestCase', 'input expected_output ksize strides')

        test_cases = [
            TestCase(
                input=[
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                expected_output=[
                    [5, 7],
                    [13, 15],
                ],
                ksize=2,
                strides=2,
            ),
            TestCase(
                input=[
                    [0, 1],
                    [4, 5],
                    [8, 9],
                    [12, 13],
                ],
                expected_output=[
                    [5],
                    [13],
                ],
                ksize=2,
                strides=2,
            ),
            TestCase(
                input=[
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                expected_output=[
                    [10, 11],
                    [14, 15],
                ],
                ksize=3,
                strides=1,
            ),
            TestCase(
                input=[
                    [0, 1, 2],
                    [4, 5, 6],
                    [8, 9, 10],
                ],
                expected_output=[
                    [5, 6],
                    [9, 10],
                ],
                ksize=2,
                strides=2,
            ),
            TestCase(
                input=[
                    [0, 1],
                    [4, 5],
                ],
                expected_output=[
                    [5],
                ],
                ksize=3,
                strides=1,
            ),
        ]

        for test_case in test_cases:
            is_correct = False
            try:
                result = max_pool1d_single(np.array(test_case.input), test_case.ksize, test_case.strides)
                is_correct = np.array_equal(result, np.array(test_case.expected_output))
            finally:
                if not is_correct:
                    max_pool1d_single(np.array(test_case.input), test_case.ksize, test_case.strides, debug=True)

            self.assertTrue(is_correct)

    def test_max_pool(self):
        TestCase = namedtuple('TestCase', 'input expected_output ksize strides')

        test_cases = [
            TestCase(
                input=[
                    [[0], [1], [2], [3]],
                    [[4], [5], [6], [7]],
                    [[8], [9], [10], [11]],
                    [[12], [13], [14], [15]],
                ],
                expected_output=[
                    [[5], [7]],
                    [[13], [15]],
                ],
                ksize=2,
                strides=2,
            ),
            TestCase(
                input=[
                    [[0, 15], [1, 14], [2, 13], [3, 12]],
                    [[4, 11], [5, 10], [6, 9], [7, 8]],
                    [[8, 7], [9, 6], [10, 5], [11, 4]],
                    [[12, 3], [13, 2], [14, 1], [15, 0]],
                ],
                expected_output=[
                    [[5, 15], [7, 13]],
                    [[13, 7], [15, 5]],
                ],
                ksize=2,
                strides=2,
            ),
        ]
        for test_case in test_cases:
            mpool = max_pool(np.array(test_case.input).shape, test_case.ksize, test_case.strides)

            result = mpool.max_pool(np.array(test_case.input))
            expected_result = np.array(test_case.expected_output)

            # Make sure max pooling is correct
            self.assertTrue(np.array_equal(result, expected_result))

            # Make sure .out() predicts output shape correctly
            self.assertTrue(np.array_equal(mpool.out(), result.shape))


def test_fc_or():
    TrainData = namedtuple('TrainData', 'inputs expected_output')

    test_data = [
        TrainData(
            inputs=[[0], [0]],
            expected_output=[0, 0]
        ),
        TrainData(
            inputs=[[0], [1]],
            expected_output=[0, 1]
        ),
        TrainData(
            inputs=[[1], [0]],
            expected_output=[0, 1]
        ),
        TrainData(
            inputs=[[1], [1]],
            expected_output=[1, 1]
        )
    ]

    fc = fully_connected_layer(2, 2)

    epochs = 1000
    for i in range(epochs):
        if i % 100 == 0:
            print(('%.1f' % (i * 100 / epochs)) + '% done')

        for data in test_data:
            output, _ = fc.feed_forward(np.array(data.inputs))

            if i == epochs - 1:
                print('---------')
                print('Input: ', data.inputs)
                print('AND: %.3f' % output[0])
                print('OR: %.3f' % output[1])

            fc.back_prop(np.array(data.expected_output), output)

if __name__ == '__main__':
    # unittest.main()
    test_fc_or()
