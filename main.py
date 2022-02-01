import math
import random
import unittest

from activation_functions import *


class Perceptron:

    def __init__(self, bias, threshold, activation_function,weights, inputs=None):
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

            print(inputs)


# print(network.feed_forward())
# print(network.perceptron_layers)
# print(Perceptron(bias=0, threshold=2, activation_function=binary_threshold))

layer1 = PerceptronLayer([Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold,
                                     weights=[-1, -1], inputs=[1, 1])])

layer2 = PerceptronLayer(
    [Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1]) for _ in range(2)])

layer3 = PerceptronLayer([Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1])])

network = PerceptronNetwork([layer1, layer2, layer3])

print(network.feed_forward())
