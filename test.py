import unittest
import tensorflow as tf
import numpy as np
from max_pool import max_pool, max_pool1d_single
from flatten import flatten

from collections import namedtuple


class TestMaxPool(unittest.TestCase):
    # arrays_to_test = [
    #     [
    #         [
    #             [[0], [1], [2], [3]],
    #             [[4], [5], [6], [7]],
    #             [[8], [9], [10], [11]]
    #         ],
    #         [
    #             [9, 10, 11],
    #             [12, 13, 14],
    #             [15, 16, 17]
    #         ],
    #     ]
    # ]

    arrays_to_test = []

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
                result = flatten(np.array(test_case.input))
                is_correct = np.array_equal(result, np.array(test_case.expected_output))
            finally:
                if not is_correct:
                    flatten(np.array(test_case.input), debug=True)

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
        for arr in self.arrays_to_test:
            arr = np.array(arr)
            tf_result = tf.nn.max_pool1d(arr, 2, 1, 'VALID')
            my_result = max_pool(arr, 2, 1)

            print(tf_result)
            print(my_result)
            self.assertTrue(np.array_equal(tf_result, my_result))


if __name__ == '__main__':
    unittest.main()
