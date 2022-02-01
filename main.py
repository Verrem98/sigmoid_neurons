import math
import random
import unittest

from activation_functions import *


class Perceptron:

    def __init__(self, bias, threshold, activation_function, weights, inputs=None):
        # [random.random() for _ in range(len(self.inputs))]
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.threshold = threshold

    def calculate_output(self):
        return self.activation_function(sum([w * i for w, i in zip(self.weights, self.inputs)]) + self.bias,
                                        self.threshold)

    def __str__(self):
        return f"{self.inputs=} | {self.weights=} | {self.bias=} | {self.threshold=} | {self.activation_function=}"


class PerceptronLayer:

    def __init__(self, perceptrons):
        self.perceptrons = perceptrons

    def get_outputs(self):
        return [p.calculate_output() for p in self.perceptrons]

    def update_inputs(self, inputs):
        for p in self.perceptrons:
            p.inputs = inputs


class PerceptronNetwork:

    def __init__(self, perceptron_layers):
        self.perceptron_layers = perceptron_layers

    def feed_forward(self):
        inputs = None
        for count, l in enumerate(self.perceptron_layers):
            if count == 0:
                inputs = l.get_outputs()
            else:
                l.update_inputs(inputs)
                inputs = l.get_outputs()

        return inputs


inputs = [0, 0]


# we consider the first nand gate as part of the input
input_perceptron = Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold,
                              weights=[-1, -1], inputs=inputs).calculate_output()

layer1 = PerceptronLayer(
    [Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1],
                inputs=[inputs[0] + input_perceptron]),
     Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1],
                inputs=[inputs[1] + input_perceptron])])

layer2 = PerceptronLayer([Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1])])

xor_network = PerceptronNetwork([layer1, layer2])

print(xor_network.feed_forward())
