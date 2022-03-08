import random


class Neuron:

    def __init__(self, bias, activation_function, nr_of_inputs):

        self.nr_of_inputs = nr_of_inputs
        self.bias = bias
        self.activation_function = activation_function
        self.inputs = None
        self.weights = None

    def calculate_output(self):
        """
        calculate the output of a Neuron by taking the sum of the product of every weight and input,
        adding the bias, and putting it into the activation function

        :return: the output of a specific Neuron
        """
        return self.activation_function(sum([w * i for w, i in zip(self.weights, self.inputs)]) + self.bias)

    def set_seed(self, seed):
        random.seed(seed)

    def predict(self, inputs):
        """
        get a prediction with a set of inputs

        :param inputs: inputs the model uses to make a prediction
        :return: the prediction
        """
        self.inputs = inputs

        return self.calculate_output()

    def randomize_weights(self):
        """
        fill the weights with random numbers between 0 and 1
        """

        self.weights = [random.uniform(-0.01, 0.01) for _ in range(self.nr_of_inputs)]

    def update(self, train_inputs, targets, lr=0.1, verbose=False):

        """
        updates the weights and bias of a Neuron based on a cost function

        :param train_inputs: the inputs we train the Neuron on
        :param targets: the targets the Neuron needs to predict
        :param lr: the learning rate
        :param verbose: print every step
        """

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
        """
        calculate the MSE of the training sets

        :param errors: list that contains the errors for each training set (difference between predicted and wanted outcame)
        :return: MSE
        """

        return sum([error ** 2 for error in errors]) / len(errors)

    def __str__(self):
        return f"{self.inputs=} | {self.weights=} | {self.bias=} | {self.activation_function=}"


class NeuronLayer:

    def __init__(self, neurons):
        self.neurons = neurons

    def get_outputs(self):
        """
        make a list of all the Neuron outputs of a specific NeuronLayer

        :return: a list of all the outputs of a NeuronLayer
        """
        return [p.calculate_output() for p in self.neurons]

    def update_inputs(self, inputs):
        """
        if the inputs of a Neuron in the layer haven't been set yet,
        set them as the outputs of the previous layer

        :param inputs: outputs from the previous layer
        """
        for p in self.neurons:
            if p.inputs is None:
                p.inputs = inputs


class NeuronNetwork:

    def __init__(self, neuron_layers):
        self.neuron_layers = neuron_layers

    def feed_forward(self):
        """
        loop through the network and return the last output

        :return: the network output
        """

        inputs = None
        for count, l in enumerate(self.neuron_layers):
            if count == 0:
                inputs = l.get_outputs()
            else:
                layer_input = inputs
                for neuron in l.neurons:
                    neuron.inputs = layer_input
                    inputs = l.get_outputs()
        return inputs

    def set_first_inputs(self, inputs):
        for neuron in self.neuron_layers[0].neurons:
            neuron.inputs = inputs


    def __str__(self):
        s = ""

        for count, nl in enumerate(self.neuron_layers):
            s += f"neuron layer {count}: {len(nl.neurons)} neurons\n"

        return s
