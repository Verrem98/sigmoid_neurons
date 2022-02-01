import unittest
from main import *
from itertools import product


class Test(unittest.TestCase):

    def test_AND(self):

        combs = list(product([0, 1], repeat=2))
        results = [Perceptron(bias=0, threshold=2, activation_function=binary_threshold, inputs=i,
                              weights=[1, 1]).calculate_output() for i in combs]
        comb_results = list(zip(combs, results))

        print(f"'AND:'{comb_results}")

        for c, r in comb_results:
            if c == (1, 1):
                self.assertEqual(r, 1, "Should be 1")
            else:
                self.assertEqual(r, 0, "Should be 0")

    def test_OR(self):

        combs = list(product([0, 1], repeat=2))
        results = [Perceptron(bias=0, threshold=1, activation_function=binary_threshold, inputs=i,
                              weights=[1, 1]).calculate_output() for i in combs]
        comb_results = list(zip(combs, results))
        print(f"'OR:'{comb_results}")

        for c, r in comb_results:
            if c != (0, 0):
                self.assertEqual(r, 1, "Should be 1")
            else:
                self.assertEqual(r, 0, "Should be 0")

    def test_INVERT(self):
        combs = [[0], [1]]
        results = [Perceptron(bias=0, threshold=-0.5, activation_function=binary_threshold, inputs=i,
                              weights=[-1]).calculate_output() for i in combs]
        comb_results = list(zip(combs, results))
        print(f"'INVERT:'{comb_results}")

        for c, r in comb_results:
            if c == [1]:
                self.assertEqual(r, 0, "Should be 0")
            else:
                self.assertEqual(r, 1, "Should be 1")

    def test_NOR(self):

        combs = list(product([0, 1], repeat=3))

        results = ([Perceptron(bias=0, threshold=0, activation_function=binary_threshold,
                               inputs=i,
                               weights=[-1, -1, -1]).calculate_output() for i in combs])

        comb_results = list(zip(combs, results))
        print(f"'NOR:'{comb_results}")

        for c, r in comb_results:

            if c != (0, 0, 0):
                self.assertEqual(r, 0, 'should be 0')
            else:
                self.assertEqual(r, 1, 'should be 1')

    def test_big_perceptron(self):
        # 3.4 + ((1*0.4)+(3*0.7)+(7*1)+(2*0.2)+(8*0.32)) = 15.86
        # we compare different thresholds to see if it works properly:
        bigceptron1 = Perceptron(bias=3.4, threshold=15.87, activation_function=binary_threshold,
                                 inputs=[1, 3, 7, 2, 8],
                                 weights=[0.4, 0.7, 1, 0.2, 0.32])

        bigceptron2 = Perceptron(bias=3.4, threshold=15.86, activation_function=binary_threshold,
                                 inputs=[1, 3, 7, 2, 8],
                                 weights=[0.4, 0.7, 1, 0.2, 0.32])

        bigceptron3 = Perceptron(bias=3.4, threshold=1, activation_function=binary_threshold,
                                 inputs=[1, 3, 7, 2, 8],
                                 weights=[0.4, 0.7, 1, 0.2, 0.32])

        self.assertEqual(bigceptron1.calculate_output(), 0, 'should be 0')
        self.assertEqual(bigceptron2.calculate_output(), 1, 'should be 1')
        self.assertEqual(bigceptron3.calculate_output(), 1, 'should be 1')


if __name__ == '__main__':
    unittest.main()
