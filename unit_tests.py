import unittest
from main import *
from itertools import product


class Test(unittest.TestCase):
    """
    ALL TEST CASES WERE BASED ON TRUTH TABLES FOUND ON WIKIPEDIA FOR THE RELEVANT LOGIC GATE
    """

    def test_AND(self):

        combs = list(product([0, 1], repeat=2))
        results = [Perceptron(bias=-2, activation_function=binary_threshold, inputs=i,
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
        results = [Perceptron(bias=-1, activation_function=binary_threshold, inputs=i,
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
        results = [Perceptron(bias=0.5, activation_function=binary_threshold, inputs=i,
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

        results = ([Perceptron(bias=0, activation_function=binary_threshold,
                               inputs=i,
                               weights=[-1, -1, -1]).calculate_output() for i in combs])

        comb_results = list(zip(combs, results))
        print(f"'NOR:'{comb_results}")

        for c, r in comb_results:

            if c != (0, 0, 0):
                self.assertEqual(r, 0, 'should be 0')
            else:
                self.assertEqual(r, 1, 'should be 1')

    def test_xor_network(self):

        combs = list(product([0, 1], repeat=2))
        results = ([create_xor_network(i).feed_forward()[0] for i in combs])
        comb_results = list(zip(combs, results))
        print(f"'XOR:'{comb_results}")

        for c, r in comb_results:
            if c[0] == c[1]:
                self.assertEqual(r, 0, 'should be 0')
            else:
                self.assertEqual(r, 1, 'should be 1')

    def test_half_adder_network(self):

        combs = list(product([0, 1], repeat=2))
        results = ([create_half_adder_network(i).feed_forward() for i in combs])
        comb_results = list(zip(combs, results))
        print(f"'Half Adder:'{comb_results}")

        for c, r in comb_results:
            if c == (0, 0):
                self.assertEqual(r, [0, 0], 'should be [0,0]')
            if c == (1, 0) or c == (0, 1):
                self.assertEqual(r, [0, 1], 'should be [0,1]')
            if c == (1, 1):
                self.assertEqual(r, [1, 0], 'should be [1,0]')


if __name__ == '__main__':
    unittest.main()
