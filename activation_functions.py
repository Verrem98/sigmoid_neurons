import math


def sigmoid(x):
    return 1 / (1 + (math.e ** -x))


def relu(x):
    return max([0, x])


#def binary_threshold(x, threshold=0):
#   return 1 if x >= threshold else 0


def binary_threshold(x):
    return 1 if x >= 0 else 0


def dummy(x):
    return x
