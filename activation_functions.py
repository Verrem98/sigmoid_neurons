import math


def sigmoid(x, args):
    return 1 / (1 + (math.e ** -x))


def relu(x, args):
    return max([0, x])


def binary_threshold(x, threshold=0):
    return 1 if x >= threshold else 0
