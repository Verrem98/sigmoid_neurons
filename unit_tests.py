import unittest
from neural_network import *
from itertools import product
import numpy as np


class Test(unittest.TestCase):
    """
    ALL TEST CASES WERE BASED ON TRUTH TABLES FOUND ON WIKIPEDIA FOR THE RELEVANT LOGIC GATE
    """

    def test_AND_learning(self):

        # we need to give our perceptron inputs so that it knows how many random weights it has to generate,
        # these do not matter
        perc = Perceptron(bias=0, activation_function=binary_threshold, nr_of_inputs=2)
        perc.randomize_weights()

        # we train the perceptron to behave like an AND gate
        perc.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1, 0, 0, 0],
                    verbose=False)

        outputs = []
        combinations = list(product([0, 1], repeat=2))
        for i in combinations:
            output = perc.predict(i)
            outputs.append(output)

            if i == (1, 1):
                self.assertEqual(output, 1, "should be 1")
            else:
                self.assertEqual(output, 0, "should be 0")

        print(f"'AND(trained)'{list(zip(combinations, outputs))}")

    def test_XOR_learning(self):
        None

        # XOR gates are not linearly sepperable, this will not work




if __name__ == '__main__':
    unittest.main()
