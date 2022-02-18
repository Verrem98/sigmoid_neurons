import random

from activation_functions import *


class Perceptron:

    def __init__(self, bias, activation_function, weights=None, inputs=None):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def calculate_output(self):
        """
        calculate the output of a perceptron by taking the sum of the product of every weight and input,
        adding the bias, and putting it into the activation function

        :return: the output of a specific perceptron
        """
        return self.activation_function(sum([w * i for w, i in zip(self.weights, self.inputs)]) + self.bias)

    def randomize_weights(self):
        """
        fill the weights with random numbers between 0 and 1
        """

        self.weights = [random.random() for _ in range(len(self.inputs))]

    def __str__(self):
        return f"{self.inputs=} | {self.weights=} | {self.bias=} | {self.threshold=} | {self.activation_function=}"


class PerceptronLayer:

    def __init__(self, perceptrons):
        self.perceptrons = perceptrons

    def get_outputs(self):
        """
        make a list of all the perceptron outputs of a specific PerceptronLayer

        :return: a list of all the outputs of a PerceptronLayer
        """
        return [p.calculate_output() for p in self.perceptrons]

    def update_inputs(self, inputs):
        """
        if the inputs of a perceptron in the layer haven't been set yet,
        set them as the outputs of the previous layer

        :param inputs: outputs from the previous layer
        """
        for p in self.perceptrons:
            if p.inputs is None:
                p.inputs = inputs


class PerceptronNetwork:

    def __init__(self, perceptron_layers):
        self.perceptron_layers = perceptron_layers

    def feed_forward(self):
        """
        loop through the network and return the last output

        :return: the network output
        """
        inputs = None
        for count, l in enumerate(self.perceptron_layers):
            if count == 0:
                inputs = l.get_outputs()
            else:
                l.update_inputs(inputs)
                inputs = l.get_outputs()

        return inputs


def create_xor_network(inputs):
    """
    create a xor-network with two inputs

    :param inputs: inputs (the input layer)
    :return: PerceptronNetwork
    """


    layer1 = PerceptronLayer(
        [Perceptron(bias=-1, activation_function=binary_threshold, weights=[1,1],
                    inputs=inputs),
         Perceptron(bias=1, activation_function=binary_threshold, weights=[-1, -1],
                    inputs=inputs)])

    layer2 = PerceptronLayer(
        [Perceptron(bias=-2, activation_function=binary_threshold, weights=[1,1])])

    return PerceptronNetwork([layer1, layer2])




def create_half_adder_network(inputs):
    """
       create a half-adder-network with two inputs

       :param inputs: inputs (the input layer)
       :return: PerceptronNetwork
       """
    layer1 = PerceptronLayer(
        [Perceptron(bias=-2, activation_function=binary_threshold, weights=[1, 1],
                    inputs=inputs),
         Perceptron(bias=-1, activation_function=binary_threshold, weights=[1, 1],
                    inputs=inputs),
         Perceptron(bias=1, activation_function=binary_threshold, weights=[-1, -1],
                    inputs=inputs)])

    layer2 = PerceptronLayer(
        [Perceptron(bias=-1, activation_function=binary_threshold, weights=[1, 0, 0]),
         Perceptron(bias=-2, activation_function=binary_threshold, weights=[0, 1, 1])])

    return PerceptronNetwork([layer1, layer2])
