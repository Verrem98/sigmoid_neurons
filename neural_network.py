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

        self.weights = [random.uniform(-0.01, 0.01) for _ in range(len(self.inputs))]

    def update(self, train_inputs, targets, lr=0.1, verbose=False):

        while True:
            errors = []
            for input, target in list(zip(train_inputs, targets)):
                self.inputs = input

                error = target - self.calculate_output()
                weight_deltas = [lr * error * inp for inp in input]
                bias_delta = lr * error
                self.bias += bias_delta
                self.weights = [sum(x) for x in list(zip(weight_deltas, self.weights))]
                errors.append(error)

                if verbose:
                    print(
                        f"{self.inputs=} | {error=} | {self.calculate_output()=} | {self.weights=} | {weight_deltas=} | {self.bias=}")
            if self.calculate_MSE(errors) == 0:
                return

    def calculate_MSE(self, errors):

        return sum([error ** 2 for error in errors]) / len(errors)

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


inputs = [1, 1]
perc = Perceptron(bias=0, activation_function=binary_threshold, inputs=inputs)
perc.randomize_weights()

#perc.update(train_inputs=[[1, 1], [5, 5], [-1, 7], [2, 19], [-4, 3], [-23, 4], [-1, -1]], targets=[1, 1, 0, 1, 0, 0, 0],
#            verbose=True)

perc.update(train_inputs=[[1, 1], [1, 0], [0, 1], [0, 0]], targets=[1,0,0,0],
           verbose=True)
perc.inputs = [0, 1]
print(perc.calculate_output())
