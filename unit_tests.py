import unittest
from neural_network import NeuronNetwork, NeuronLayer, Neuron
from activation_functions import binary_threshold, sigmoid, rounded_sigmoid
from itertools import product
import numpy as np


class Test(unittest.TestCase):
    """
    ALL TEST CASES WERE BASED ON TRUTH TABLES FOUND ON WIKIPEDIA FOR THE RELEVANT LOGIC GATE
    """

    def test_AND_OR_INVERT_SAME_INPUTS(self):

        combs = list(product([0, 1], repeat=2))

        print(
            "it looks like using perceptron parameters doesn't work for the sigmoid function for AND and OR\nbut it does seem to work for INVERT: ")
        print("AND(p1 parameters): ")
        for c in combs:
            neuron_AND = Neuron(bias=-2, activation_function=rounded_sigmoid, nr_of_inputs=2)
            neuron_AND.weights = [1, 1]
            neuron_AND.inputs = c
            print(c, neuron_AND.calculate_output())

        print("OR(p1 parameters): ")
        for c in combs:
            neuron_OR = Neuron(bias=-1, activation_function=rounded_sigmoid, nr_of_inputs=2)
            neuron_OR.weights = [1, 1]
            neuron_OR.inputs = c
            print(c, neuron_OR.calculate_output())

        combs = [[1], [0]]
        print("INVERT(p1 parameters): ")

        for c in combs:
            neuron_INVERT = Neuron(bias=0.5, activation_function=rounded_sigmoid, nr_of_inputs=1)
            neuron_INVERT.weights = [-1]
            neuron_INVERT.inputs = c
            print(c, neuron_INVERT.calculate_output())

    def test_INVERT_neuron(self):

        # we need to give our perceptron the number of inputs so that it knows how many random weights it has to generate
        perc = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=1)
        perc.randomize_weights()

        # we train the perceptron to behave like an AND gate
        perc.update(train_inputs=[[1], [0]], targets=[0, 1],
                    verbose=False)

        outputs = []
        combinations = [[1], [0]]
        for i in combinations:
            output = perc.predict(i)
            outputs.append(output)

            if i == [0]:
                self.assertEqual(output, 1, "should be 1")
            if 1 == [1]:
                self.assertEqual(output, 0, "should be 0")

        print(f"'INVERT(trained)'{list(zip(combinations, outputs))}, weights: {perc.weights}, bias: {perc.bias}")

    def test_AND_neuron(self):

        # we need to give our perceptron the number of inputs so that it knows how many random weights it has to generate
        perc = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
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

        print(f"'AND(trained)'{list(zip(combinations, outputs))}, weights: {perc.weights}, bias: {perc.bias}")

    def test_or_neuron(self):

        # we need to give our perceptron the number of inputs so that it knows how many random weights it has to generate
        perc = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        perc.randomize_weights()

        perc.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1, 1, 1, 0],
                    verbose=False)

        outputs = []
        combinations = list(product([0, 1], repeat=2))
        for i in combinations:
            output = perc.predict(i)
            outputs.append(output)

            if 1 in i:
                self.assertEqual(output, 1, "should be 1")
            else:
                self.assertEqual(output, 0, "should be 0")

        print(f"'OR(trained)'{list(zip(combinations, outputs))}, weights: {perc.weights}, bias: {perc.bias}")

    def test_XOR_learning(self):
        None

        # XOR gates are not linearly sepperable, this will not work

    def test_NAND_learning(self):

        # we need to give our perceptron the number of inputs so that it knows how many random weights it has to generate
        perc = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        perc.randomize_weights()

        perc.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[0, 1, 1, 1],
                    verbose=False)

        outputs = []
        combinations = list(product([0, 1], repeat=2))
        for i in combinations:
            output = perc.predict(i)
            outputs.append(output)

            if i == (1, 1):
                self.assertEqual(output, 0, "should be 0")
            else:
                self.assertEqual(output, 1, "should be 1")

        print(f"'NAND(trained)'{list(zip(combinations, outputs))}, weights: {perc.weights}, bias: {perc.bias}")

    def test_NOR_learning(self):

        combs = list(product([0, 1], repeat=3))

        perc = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=3)
        perc.randomize_weights()

        perc.update(train_inputs=combs, targets=[1] + [0 for _ in range(len(combs) - 1)],
                    verbose=False)

        outputs = []
        for i in combs:
            output = perc.predict(i)
            outputs.append(output)

            if i == (0, 0, 0):
                self.assertEqual(output, 1, "should be 1")
            else:
                self.assertEqual(output, 0, "should be 0")

        print(f"'NOR(trained)'{list(zip(combs, outputs))}, weights: {perc.weights}, bias: {perc.bias}")

    def test_neuron_XOR(self):

        or_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        or_neuron.randomize_weights()
        or_neuron.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1, 1, 1, 0],
                         verbose=False)

        nand_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        nand_neuron.randomize_weights()
        nand_neuron.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[0, 1, 1, 1],
                           verbose=False)

        and_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        and_neuron.randomize_weights()
        and_neuron.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1, 0, 0, 0],
                          verbose=False)

        combinations = list(product([0, 1], repeat=2))

        outputs = []
        for c in combinations:

            l1 = NeuronLayer([or_neuron, nand_neuron])
            l2 = NeuronLayer([and_neuron])
            nw = NeuronNetwork([l1, l2])

            # not sure if setting the inputs of the first hidden layer should be part of the feed forward function, please let me know if that is the case
            # for now it is a function of the NeuronNetwork class
            nw.set_first_inputs(inputs=c)

            output = nw.feed_forward()[0]
            outputs.append(output)

            if c.count(1) == 1:
                self.assertEqual(output, 1, "should be 1")
            else:
                self.assertEqual(output, 0, "should be 0")

        print(f"'XOR(trained)'{list(zip(combinations, outputs))}")

    def test_neuron_half_adder(self):

        or_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        or_neuron.randomize_weights()
        or_neuron.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1, 1, 1, 0],
                         verbose=False)

        nand_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        nand_neuron.randomize_weights()
        nand_neuron.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[0, 1, 1, 1],
                           verbose=False)

        and_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=2)
        and_neuron.randomize_weights()
        and_neuron.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1, 0, 0, 0],
                          verbose=False)

        # we can still train this neuron on two inputs only, because the third is irrelevant
        sum_and_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=3)
        sum_and_neuron.randomize_weights()
        sum_and_neuron.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1, 0, 0, 0],
                              verbose=False)

        # this neuron just looks at the o or 1 from the AND neuron in the first layer
        carry_and_neuron = Neuron(bias=0, activation_function=rounded_sigmoid, nr_of_inputs=3)
        carry_and_neuron.randomize_weights()
        carry_and_neuron.update(
            train_inputs=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            targets=[0, 1, 0, 1, 0, 1, 0, 1],
            verbose=False)

        combinations = list(product([0, 1], repeat=2))

        outputs = []
        for c in combinations:

            l1 = NeuronLayer([or_neuron, nand_neuron, and_neuron])
            l2 = NeuronLayer([sum_and_neuron, carry_and_neuron])
            nw = NeuronNetwork([l1, l2])

            # not sure if setting the inputs of the first hidden layer should be part of the feed forward function, please let me know if that is the case
            # for now it is a function of the NeuronNetwork class
            nw.set_first_inputs(inputs=c)

            output = nw.feed_forward()
            outputs.append(output)

            if c == (0, 0):
                self.assertEqual(output, [0, 0], "should be [0,0]")
            elif c == (0, 1) or c == (1, 0):
                self.assertEqual(output, [1, 0], "should be [1,0]")

            else:
                self.assertEqual(output, [0, 1], "should be [0,1]")

        print(f"half_adder(trained)'{list(zip(combinations, outputs))}")


if __name__ == '__main__':
    unittest.main()
