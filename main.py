from activation_functions import *

class Perceptron:

    def __init__(self, bias, threshold, activation_function, weights, inputs=None):

        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.threshold = threshold

    def calculate_output(self):
        """
        calculate the output of a perceptron by taking the sum of the product of every weight and input,
        adding the bias, and putting it into the activation function

        :return: the output of a specific perceptron
        """
        return self.activation_function(sum([w * i for w, i in zip(self.weights, self.inputs)]) + self.bias,
                                        self.threshold)

    def __str__(self):
        return f"{self.inputs=} | {self.weights=} | {self.bias=} | {self.threshold=} | {self.activation_function=}"


class PerceptronLayer:

    def __init__(self, perceptrons):
        self.perceptrons = perceptrons

    def get_outputs(self):
        """
        make a list of all the perceptron outputs of a specific PercentronLayer

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

    :param inputs: inputs
    :return: PerceptronNetwork
    """
    # we consider the first nand gate as part of the input
    input_perceptron = Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold,
                                  weights=[-1, -1], inputs=inputs).calculate_output()

    layer1 = PerceptronLayer(
        [Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1],
                    inputs=[inputs[0] + input_perceptron]),
         Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1],
                    inputs=[inputs[1] + input_perceptron])])

    layer2 = PerceptronLayer(
        [Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1])])

    return PerceptronNetwork([layer1, layer2])


def create_half_adder_network(inputs):
    """
        create a half-adder-network with two inputs

        :param inputs: inputs
        :return: PerceptronNetwork
        """
    # the only difference compared to the xor network is an additional AND-gate with the original inputs
    # in layer2

    # we consider the first nand gate as part of the input
    input_perceptron = Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold,
                                  weights=[-1, -1], inputs=inputs)

    layer1 = PerceptronLayer(
        [Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1],
                    inputs=[inputs[0] + input_perceptron.calculate_output()]),
         Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1],
                    inputs=[inputs[1] + input_perceptron.calculate_output()])])

    layer2 = PerceptronLayer(
        [Perceptron(bias=0, threshold=-1.99, activation_function=binary_threshold, weights=[-1, -1]),
         Perceptron(bias=0, threshold=2, activation_function=binary_threshold, weights=[1, 1], inputs=inputs)])

    return PerceptronNetwork([layer1, layer2])